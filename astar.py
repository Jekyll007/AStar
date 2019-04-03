import heapq
import math

def Heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)
# def Heuristic(goal, next, current):
#     (x1, y1) = goal
#     (x2, y2) = next
#     heuristic = abs(x1 - x2) + abs(y1 - y2)
#     dx1 = current[0] - next[0]
#     dy1 = current[1] - next[1]
# #     dx2 = start[0] - next[0]
# #     dy2 = start[1] - next[1]
# #     cross = abs(dx1*dy2 - dx2*dy1)
# #     heuristic += cross*0.001
#     D = math.sqrt(2) - 2
#     heuristic = (dx1 + dy1) + D * min(dx1, dy1) 
#     return heuristic
        
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]
    
class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1), (x-1, y+1), (x+1, y+1), (x-1, y-1), (x+1, y-1)]
        if (x + y) % 2 == 0: results.reverse()
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}
    
    def cost(self, from_node, to_node):
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        if dx != 0 and dy != 0:
            return 14
        else:
            return 10
#         return self.weights.get(to_node, 1)


def AStarSearch(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        for next in graph.neighbors(current):              
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + Heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
        
    return came_from, cost_so_far
        
def ReconstructPath(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path
        
        
