from epdserve.orchestrator import *
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load estimator updates data after this many seconds
LOAD_ESTIMATOR_EVENT_LOOP = 1

# Resource allocation takes decisions after this many seconds
RESOURCE_ALLOCATOR_EVENT_LOOP = 5


class LoadEstimator():
    def __init__(self, encoding_cluster, context_cluster, decoding_cluster):
        self.encoding_cluster= encoding_cluster
        self.context_cluster = context_cluster
        self.decoding_cluster = decoding_cluster
        self.timing_data = deque(maxlen=20)
        self.block_data = deque(maxlen=20)
        self.queue_data = deque(maxlen=20)
        self.gpu_data = deque(maxlen=20)
        self.agg_metrics = []

    def add_timing_data(self, data):
        computed_times = {
            'encoding_time': data['encoding_end'].timestamp - data['encoding_begin'].timestamp,
            'encoding_migration_time': data['encoding_migration_end'].timestamp - data['encoding_migration_begin'].timestamp,
            'context_time': data['context_end'].timestamp - data['context_begin'].timestamp,
            'decoding_time': data['decoding_end'].timestamp - data['decoding_begin'].timestamp
        }
        self.timing_data.append(computed_times)

    def add_queue_data(self, data):
        self.queue_data.append(data)
        pass

    def add_block_data(self, data):
        self.block_data.append(data)
        pass

    def add_gpu_data(self, data):
        self.gpu_data.append(data)
        pass

    def update_queues(self, gpu_data=None):
        block_keys = ['A.blocks%/A.encoding', 'A.blocks%/B.context', 'A.blocks%/C.decoding']
        queue_keys = ['B.queuing/A.encoding#', 'B.queuing/B.context#', 'B.queuing/C.decoding#', 'C.running/A.encoding#', 'C.running/B.context#', 'C.running/C.decoding#', 'D.awaiting/A.encoding#', 'D.awaiting/B.context#', 'D.awaiting/C.decoding#', 'E.exited/A.encoding#', 'E.exited/B.context#', 'E.exited/C.decoding#']
        block_dict = {key: self.agg_metrics[-1][key] for key in block_keys if key in self.agg_metrics[-1]}
        queue_dict = {key: self.agg_metrics[-1][key] for key in queue_keys if key in self.agg_metrics[-1]}
        self.add_block_data(block_dict)
        self.add_queue_data(queue_dict)
        # self.add_gpu_data(gpu_data)

    def get_cluster_status(self):
        block_usage_index = []
        scheduler_usage_index = []
        encoding_scheduler_usage = []
        encoding_block_usage = []
        for dp_rank, engine in enumerate(self.encoding_cluster.engines):
            encoding_scheduler_usage.append(engine.get_scheduler_status())
            encoding_block_usage.append(engine.get_block_status())
            scheduler_usage_index.append(f'encoding(DP{dp_rank})')
            block_usage_index.append(f'encoding(DP{dp_rank})')

        context_scheduler_usage = []
        context_block_usage = []
        for dp_rank, engine in enumerate(self.context_cluster.engines):
            context_scheduler_usage.append(engine.get_scheduler_status())
            context_block_usage.append(engine.get_block_status())
            scheduler_usage_index.append(f'context(DP{dp_rank})')
            block_usage_index.append(f'context(DP{dp_rank})')
            context_block_usage.append(engine.get_block_status(vision=True))
            block_usage_index.append(f'  conviz(DP{dp_rank})')

        decoding_scheduler_usage = []
        decoding_block_usage = []
        for dp_rank, engine in enumerate(self.decoding_cluster.engines):
            decoding_scheduler_usage.append(engine.get_scheduler_status())
            decoding_block_usage.append(engine.get_block_status())
            scheduler_usage_index.append(f'decoding(DP{dp_rank})')
            block_usage_index.append(f'decoding(DP{dp_rank})')
        return encoding_block_usage, context_block_usage, decoding_block_usage, block_usage_index, \
               encoding_scheduler_usage, context_scheduler_usage, decoding_scheduler_usage, scheduler_usage_index

    def get_gpu_status(self, encoding_block_usage, context_block_usage, decoding_block_usage):
        encoding_gpus = [int(block_usage['gpus']) for block_usage in encoding_block_usage]
        context_gpus = [int(block_usage['gpus']) for block_usage in context_block_usage]
        decoding_gpus = [int(block_usage['gpus']) for block_usage in decoding_block_usage]

        encoding_stats = [{
            '#gpu': f'encoding(#{device.nvml_index})',
            'gpu%': device.gpu_utilization(),
            'mem%': device.memory_percent()} for device in self.nvitop_devices if device.nvml_index in encoding_gpus]

        context_stats = [{
            '#gpu': f'context(#{device.nvml_index})',
            'gpu%': device.gpu_utilization(),
            'mem%': device.memory_percent()} for device in self.nvitop_devices if device.nvml_index in context_gpus]

        decoding_stats = [{
            '#gpu': f'decoding(#{device.nvml_index})',
            'gpu%': device.gpu_utilization(),
            'mem%': device.memory_percent()} for device in self.nvitop_devices if device.nvml_index in decoding_gpus]

        avg_stats = [{
            '#gpu': f'encoding(avg)',
            'gpu%': np.mean([stat['gpu%'] for stat in encoding_stats]),
            'mem%': np.mean([stat['mem%'] for stat in encoding_stats])
        },
        {
            '#gpu': f'context(avg)',
            'gpu%': np.mean([stat['gpu%'] for stat in context_stats]),
            'mem%': np.mean([stat['mem%'] for stat in context_stats])

        },
        {
            '#gpu': f'decoding(avg)',
            'gpu%': np.mean([stat['gpu%'] for stat in decoding_stats]),
            'mem%': np.mean([stat['mem%'] for stat in decoding_stats])
        }]

        return encoding_stats, context_stats, decoding_stats, avg_stats

    def compute_state_agg_stats(self, encoding_block_usage, context_block_usage, decoding_block_usage,  \
               encoding_scheduler_usage, context_scheduler_usage, decoding_scheduler_usage, avg_gpu_stats=None):

        context_block_usage = [block_usage for (idx, block_usage) in enumerate(context_block_usage) if idx %2 ==0]
        metrics = {}
        
        metric_id = ord('A')
        encoding_blocks_percent = np.mean([float(block_usage['gpu'].split('%')[0]) for block_usage in encoding_block_usage])
        context_blocks_percent = np.mean([float(block_usage['gpu'].split('%')[0]) for block_usage in context_block_usage])
        decoding_blocks_percent = np.mean([float(block_usage['gpu'].split('%')[0]) for block_usage in decoding_block_usage])
        metrics[f'{chr(metric_id)}.blocks%/A.encoding'] = encoding_blocks_percent
        metrics[f'{chr(metric_id)}.blocks%/B.context'] = context_blocks_percent
        metrics[f'{chr(metric_id)}.blocks%/C.decoding'] = decoding_blocks_percent

        stat_keys = ['queuing', 'running', 'awaiting', 'exited']
        for stat in stat_keys:
            metric_id+=1
            encoding_stat = np.sum([float(scheduler_usage[stat]) for scheduler_usage in encoding_scheduler_usage])
            context_stat = np.sum([float(scheduler_usage[stat]) for scheduler_usage in context_scheduler_usage])
            decoding_stat = np.sum([float(scheduler_usage[stat]) for scheduler_usage in decoding_scheduler_usage])
            metrics[f'{chr(metric_id)}.{stat}/A.encoding#'] = encoding_stat
            metrics[f'{chr(metric_id)}.{stat}/B.context#'] = context_stat
            metrics[f'{chr(metric_id)}.{stat}/C.decoding#'] = decoding_stat

        if avg_gpu_stats is not None:
            metric_id+=1
            metrics[f'{chr(metric_id)}.gpu_util/A.encoding'] = avg_gpu_stats[0]['gpu%']
            metrics[f'{chr(metric_id)}.gpu_util/B.context'] = avg_gpu_stats[1]['gpu%']
            metrics[f'{chr(metric_id)}.gpu_util/C.decoding'] = avg_gpu_stats[2]['gpu%']

        self.agg_metrics.append(metrics)

    def log_tensorboard(self, metrics):
        if not hasattr(self, 'tb_writer'):
            self.tb_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), f'../runs/{self.model_config.run_name}'))
            self.global_step = -1

        self.global_step += 1
        for tag, scalar in metrics.items():
            self.tb_writer.add_scalar(tag, scalar, global_step=self.global_step)

    def log_terminal(self, encoding_block_usage, context_block_usage, decoding_block_usage,
                     encoding_scheduler_usage, context_scheduler_usage, decoding_scheduler_usage,
                     block_usage_index, scheduler_usage_index):
        print()
        # print(pd.DataFrame(encoding_gpu_stats+context_gpu_stats+decoding_gpu_stats+avg_gpu_stats).set_index('#gpu').round(2))
        print(pd.DataFrame(encoding_block_usage + context_block_usage + decoding_block_usage, index=block_usage_index))
        print(pd.DataFrame(encoding_scheduler_usage + context_scheduler_usage + decoding_scheduler_usage, index=scheduler_usage_index))

    async def start_event_loop(self):
        while True:
            ''' Stage level queue and block stats '''
            encoding_block_usage, context_block_usage, decoding_block_usage, block_usage_index, \
            encoding_scheduler_usage, context_scheduler_usage, decoding_scheduler_usage, \
                scheduler_usage_index = self.get_cluster_status()
            
            ''' GPU stats like utlization and memory'''
            # encoding_gpu_stats, context_gpu_stats, decoding_gpu_stats, avg_gpu_stats = self.get_gpu_status(encoding_block_usage, context_block_usage, decoding_block_usage)

            ''' Compute Aggregated Stats '''
            ### TF metrics computation and logging ###
            self.compute_state_agg_stats(encoding_block_usage, context_block_usage, decoding_block_usage,\
                                        encoding_scheduler_usage, context_scheduler_usage, 
                                        decoding_scheduler_usage, avg_gpu_stats=None)

            ''' Log to terminal '''
            self.log_terminal(encoding_block_usage, context_block_usage, decoding_block_usage,
                        encoding_scheduler_usage, context_scheduler_usage, decoding_scheduler_usage,
                        block_usage_index, scheduler_usage_index)

            ''' Log to tensorboard '''
            # self.log_tensorboard(self.agg_metrics[-1])

            ''' Update queues '''
            self.update_queues()

            await asyncio.sleep(LOAD_ESTIMATOR_EVENT_LOOP)

class LoadEstimatorFromHistoryReqs():
    def __init__(self, encoding_cluster, context_cluster, decoding_cluster, window_size=100):
        self.init_config = [len(encoding_cluster.engines), len(context_cluster.engines), len(decoding_cluster.engines)] 
        self.ep_bridge=context_cluster.encode_context_bridge_queue
        self.history_data:"List[Request]" = deque(maxlen=window_size) 
        self.history_ids = set() 
    
    async def start_event_loop(self):
        while True: 
            reqs = list(self.ep_bridge._queue)
            for mreq in reqs:
                request=mreq.req
                request_id = request.req.request_id  
                if request_id not in self.history_ids:
                    if len(self.history_data) == self.history_data.maxlen:
                        oldest_request = self.history_data.popleft()  
                        self.history_ids.remove(oldest_request.req.request_id) 

                    self.history_data.append(request)
                    self.history_ids.add(request_id)
            
class ResourceAllocator():
    def __init__(self, load_estimator):
        self.load_estimator = load_estimator
        self.requested_migrations = deque()

    def dequeue(self):
        if self.requested_migrations:
            return self.requested_migrations.popleft()
        else:
            return None

    def submit_migration_request(self, item):
        self.requested_migrations.append(item)

    async def start_event_loop(self):
        while True:
            self.compute_migrations()
            await asyncio.sleep(RESOURCE_ALLOCATOR_EVENT_LOOP)

class ResourceAllocatorOnce(ResourceAllocator):
    def compute_migrations(self):
        if not hasattr(self, 'simulate_once'):
            self.simulate_once = True
            self.submit_migration_request('D->E')

class ResourceAllocatorRuntime(ResourceAllocator):
    def get_stage(self, stage_num):
        mapping = {0: 'E', 1: 'P', 2: 'D'}
        return mapping.get(stage_num, "Invalid input")
    
    def decide_migration(self, stage_queue_data, stage_engine_count):
        # Step 1: Determine the stage with the highest queue (to_stage)
        max_queue = max(stage_queue_data)
        to_stage = stage_queue_data.index(max_queue)

        # Step 2: Check if all stages have similarly high queuing (no migration needed)
        if all(queue >= max_queue / 2 for queue in stage_queue_data):
            return None  # No migration if all stages have similarly high queuing

        # Step 3: Determine the stage with the most spare engines and relatively low queue (from_stage)
        from_stage = None
        max_spare_engines = 0

        for i in range(len(stage_queue_data)):
            # Skip the to_stage when considering from_stage
            if i == to_stage:
                continue
            # Check conditions for low queue and spare engines
            if stage_queue_data[i] < max_queue and stage_engine_count[i] > 1:
                # Calculate spare engines
                spare_engines = stage_engine_count[i] - 1  # Reserve atleast 1 engine for the stage itself
                if spare_engines > max_spare_engines:
                    max_spare_engines = spare_engines
                    from_stage = i

        # Step 4: Ensure a valid from_stage is found
        if from_stage is not None:
            return (self.get_stage(from_stage), self.get_stage(to_stage))

        return None

    def compute_migrations(self):
        latest_queue_data = self.load_estimator.queue_data[-1]
        # Queuing: ['B.queuing/A.encoding#', 'B.queuing/B.context#', 'B.queuing/C.decoding#', 
        # Running: 'C.running/A.encoding#', 'C.running/B.context#', 'C.running/C.decoding#', 
        # Waiting to be transferred: 'D.awaiting/A.encoding#', 'D.awaiting/B.context#', 'D.awaiting/C.decoding#', 
        # Completed and exited the stage: 'E.exited/A.encoding#', 'E.exited/B.context#', 'E.exited/C.decoding#'])
        
        stage_queue_data = [int(latest_queue_data['B.queuing/A.encoding#']),
                            int(latest_queue_data['B.queuing/B.context#']),
                            int(latest_queue_data['B.queuing/C.decoding#'])]
        stage_engine_count = [len(self.load_estimator.encoding_cluster.engines),
                              len(self.load_estimator.context_cluster.engines),
                              len(self.load_estimator.decoding_cluster.engines),]

        result = self.decide_migration(stage_queue_data, stage_engine_count)
        if result is not None:
            self.submit_migration_request(f'{result[0]}->{result[1]}')

from scipy.interpolate import interp1d
class ResourceAllocatorSimulator(ResourceAllocator):
    def __init__(self, load_estimator):
        super().__init__(load_estimator)
        self.cur_config = self.load_estimator.init_config
        self.ngpu = sum(self.cur_config)
    
    def compute_migrations(self):
        from simulator.simulate_dist import run_experiment 
        self.load_estimator:LoadEstimatorFromHistoryReqs
        cur_workloads:"List[Request]" = list(self.load_estimator.history_data)
        min_time = min(req.arrival_time for req in cur_workloads)
        for req in cur_workloads:
            req.arrival_time -= min_time 
        best_config = None 
        best_goodput= -1
        
        epd_options = [
            (num_e, num_p, 8 - num_p - num_e) 
            for num_e in range(1, 7) 
            for num_p in range(1, 7 - num_e)
        ]
        rates = [ 0.25, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0 ]
        for dp_encode, dp_prefill, dp_decode in epd_options: 
            attainments = []
            for rate in rates:
                args = [
                        '--rate', str(rate),
                        '--N', str(len(cur_workloads)), 
                        '--dp-encode', str(dp_encode),
                        '--dp-prefill', str(dp_prefill), 
                        '--dp-decode', str(dp_decode),
                    ]
                outputs = run_experiment(args, external_workloads=cur_workloads)
                attainments.append(outputs.get('attainment', 0))
            interp_func = interp1d(attainments, rates, kind='linear', fill_value='extrapolate')
            goodput = float(interp_func(0.9))  # Interpolated rate for 90% attainment
        
            if goodput > best_goodput:
                best_goodput = goodput
                best_config = (dp_encode, dp_prefill, dp_decode)
        
        if best_config and tuple(best_config) != tuple(self.cur_config):
            while tuple(best_config) != tuple(self.cur_config):
                cur_e, cur_p, cur_d = self.cur_config
                tar_e, tar_p, tar_d = best_config

                if cur_e > tar_e and cur_p < tar_p:
                    cur_e -= 1
                    self.submit_migration_request("E->P")
                    cur_p += 1

                elif cur_e < tar_e and cur_p > tar_p:
                    cur_e += 1
                    self.submit_migration_request("P->E")
                    cur_p -= 1

                elif cur_p > tar_p and cur_d < tar_d:
                    cur_p -= 1
                    self.submit_migration_request("P->D")
                    cur_d += 1
                elif cur_p < tar_p and cur_d > tar_d:
                    cur_p += 1
                    self.submit_migration_request("D->P")
                    cur_d -= 1

                if cur_d > tar_d and cur_e < tar_e:
                    cur_d -= 1
                    self.submit_migration_request("D->E")
                    cur_e += 1
                elif cur_e > tar_e and cur_d < tar_d:
                    cur_e -= 1
                    self.submit_migration_request("E->D")
                    cur_d += 1

            self.cur_config = (cur_e, cur_p, cur_d)
                
            