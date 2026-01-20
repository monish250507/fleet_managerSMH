import math
import heapq

class PathPlanner:
    def __init__(self):
        # Define Factory Graph
        # Nodes: Zones
        # Edges: Connected paths with weights (distance, congestion)
        self.graph = {
            'A': {'X': 10},
            'B': {'X': 10},
            'C': {'X': 10},
            'D': {'X': 10},
            'X': {'A': 10, 'B': 10, 'C': 10, 'D': 10}
        }
        
        # Zone coordinates for heuristics (Euclidean distance)
        self.coordinates = {
            'A': (10, 0),
            'B': (0, 10),
            'C': (0, -10),
            'D': (-10, 0),
            'X': (0, 0)
        }
        
        # Congestion costs
        self.node_status = {
            'A': 'FREE', 'B': 'FREE', 'C': 'FREE', 'D': 'FREE', 'X': 'FREE'
        }
        self.current_traffic_state = "SAFE"

    def update_traffic_state(self, state):
        """Update global traffic state (SAFE, WARNING, CRITICAL)"""
        self.current_traffic_state = state

    def update_zone_status(self, zone, status):
        """Update specific zone status (ALLOW/BLOCK)"""
        # In our simplified model, Zone Manager sends global permission or we parse it
        # For now, we assume status is for Zone X as it's the critical resource
        if zone == 'X':
            self.node_status['X'] = status

    def get_cost(self, u, v):
        """Get cost between nodes u and v taking congestion into account"""
        base_cost = self.graph[u].get(v, float('inf'))
        
        multiplier = 1.0
        
        # If target node is blocked, cost is infinite
        if self.node_status.get(v) == 'BLOCK':
            return float('inf')
            
        # Traffic multiplier
        if self.current_traffic_state == "CRITICAL":
            multiplier = 5.0
        elif self.current_traffic_state == "WARNING":
            multiplier = 2.0
            
        return base_cost * multiplier

    def heuristic(self, u, v):
        x1, y1 = self.coordinates[u]
        x2, y2 = self.coordinates[v]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def find_path(self, start, goal):
        """A* Path Finding"""
        if start not in self.graph or goal not in self.graph:
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {node: float('inf') for node in self.graph}
        g_score[start] = 0
        
        f_score = {node: float('inf') for node in self.graph}
        f_score[start] = self.heuristic(start, goal)
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1] # Reverse
            
            for neighbor in self.graph[current]:
                tentative_g_score = g_score[current] + self.get_cost(current, neighbor)
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None # No path found
