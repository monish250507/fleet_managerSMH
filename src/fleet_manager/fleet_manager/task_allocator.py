class TaskAllocator:
    def __init__(self):
        self.agves = {
            'agv1': {'status': 'IDLE', 'position': 'A', 'current_task': None},
            'agv2': {'status': 'IDLE', 'position': 'B', 'current_task': None}
        }
        self.tasks_queue = []

    def update_agv_status(self, agv_id, status, position=None):
        if agv_id in self.agves:
            self.agves[agv_id]['status'] = status
            if position:
                self.agves[agv_id]['position'] = position

    def add_task(self, task_id, source, destination):
        task = {
            'id': task_id,
            'source': source,
            'destination': destination,
            'status': 'PENDING'
        }
        self.tasks_queue.append(task)
        return task

    def assign_tasks(self):
        assignments = []
        
        # Simple Logic: First Free AGV gets First Pending Task
        # In a real system, we would calculate cost (distance to source)
        
        for task in self.tasks_queue:
            if task['status'] == 'PENDING':
                best_agv = None
                
                # Find free AGV
                for agv_id, agv_data in self.agves.items():
                    if agv_data['status'] == 'IDLE':
                        best_agv = agv_id
                        break
                
                if best_agv:
                    task['status'] = 'ASSIGNED'
                    self.agves[best_agv]['status'] = 'BUSY'
                    self.agves[best_agv]['current_task'] = task['id']
                    
                    assignments.append({
                        'agv_id': best_agv,
                        'task': task
                    })
        
        return assignments

    def complete_task(self, agv_id):
        if agv_id in self.agves:
            self.agves[agv_id]['status'] = 'IDLE'
            self.agves[agv_id]['current_task'] = None
