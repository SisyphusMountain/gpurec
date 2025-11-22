import heapq
import random
import sys

# Increase recursion depth for deep trees
sys.setrecursionlimit(5000)

class UnrootedTreeTaskSim:
    def __init__(self, num_trees, leaves_per_tree):
        self.num_trees = num_trees
        self.leaves_per_tree = leaves_per_tree
        self.adj = {} 
        self.tasks = {} 
        self.ready_queue = [] 
        self.node_offset = 0
        
    def generate_unrooted_yule(self):
        """
        Generates a random unrooted binary tree with N leaves.
        Returns adjacency list for this specific tree.
        """
        local_adj = {0: [1], 1: [0]}
        next_leaf = 2
        next_internal = self.leaves_per_tree
        
        # Keep track of edges to split
        edges = [(0, 1)]
        
        while next_leaf < self.leaves_per_tree:
            edge_idx = random.randint(0, len(edges) - 1)
            u, v = edges[edge_idx]
            
            new_node = next_internal
            next_internal += 1
            new_leaf = next_leaf
            next_leaf += 1
            
            # Update Graph
            local_adj[u].remove(v)
            local_adj[v].remove(u)
            
            local_adj[new_node] = [u, v, new_leaf]
            local_adj[u].append(new_node)
            local_adj[v].append(new_node)
            local_adj[new_leaf] = [new_node]
            
            edges[edge_idx] = edges[-1]
            edges.pop()
            
            edges.append((u, new_node))
            edges.append((v, new_node))
            edges.append((new_node, new_leaf))
            
        return local_adj, next_internal

    def build_dependencies(self):
        print(f"Generating {self.num_trees} trees...")
        
        total_tasks = 0
        
        for t_id in range(self.num_trees):
            adj, max_node = self.generate_unrooted_yule()
            
            tree_tasks = {} 
            
            # 1. Define all Directed Edge Tasks
            for u in adj:
                for v in adj[u]:
                    task_key = (t_id, u, v)
                    tree_tasks[task_key] = {'deps': 0, 'parents': [], 'is_leaf_clade': False}

            # 2. Link Dependencies
            for u in adj:
                neighbors = adj[u]
                if len(neighbors) == 1:
                    v = neighbors[0]
                    # Leaf Clade
                    tree_tasks[(t_id, u, v)]['is_leaf_clade'] = True
                    tree_tasks[(t_id, u, v)]['height'] = 0
                else:
                    # Internal Clade
                    for v in neighbors:
                        others = [n for n in neighbors if n != v]
                        current_task = (t_id, u, v)
                        
                        for input_node in others:
                            dependency = (t_id, input_node, u)
                            tree_tasks[current_task]['deps'] += 1
                            tree_tasks[dependency]['parents'].append(current_task)

            # 3. Add Ubiquitous Clade Task
            # FIX: Use (-1, -1) instead of 'ubiq' to ensure integer comparison
            ubiq_task = (t_id, -1, -1)
            tree_tasks[ubiq_task] = {'deps': 0, 'parents': [], 'height': -1}
            
            # Pick arbitrary internal node to feed the ubiquitous clade
            root_node = self.leaves_per_tree
            
            for neighbor in adj[root_node]:
                dep = (t_id, neighbor, root_node)
                tree_tasks[ubiq_task]['deps'] += 1
                tree_tasks[dep]['parents'].append(ubiq_task)

            self.tasks.update(tree_tasks)
            total_tasks += len(tree_tasks)
            
        print(f"Graph built. Total tasks: {total_tasks}")

    def compute_heights_and_init_queue(self):
        print("Computing topological heights...")
        temp_deps = {k: v['deps'] for k, v in self.tasks.items()}
        q = [k for k, v in temp_deps.items() if v == 0]
        
        while q:
            curr = q.pop(0)
            curr_h = self.tasks[curr].get('height', 0)
            
            for p in self.tasks[curr]['parents']:
                p_h = self.tasks[p].get('height', 0)
                self.tasks[p]['height'] = max(p_h, curr_h + 1)
                
                temp_deps[p] -= 1
                if temp_deps[p] == 0:
                    q.append(p)
                    
        print("Initializing Scheduler Heap...")
        for tid, info in self.tasks.items():
            if info['deps'] == 0:
                heapq.heappush(self.ready_queue, (-info['height'], tid))

    def run_simulation(self, batch_size=1024):
        print(f"Running simulation with Batch Size {batch_size}...")
        waves = []
        
        while self.ready_queue:
            wave_count = 0
            finished_tasks = []
            
            # Fill Batch
            while self.ready_queue and wave_count < batch_size:
                prio, tid = heapq.heappop(self.ready_queue)
                finished_tasks.append(tid)
                wave_count += 1
            
            waves.append(wave_count)
            
            # Unlock dependencies
            for tid in finished_tasks:
                for p in self.tasks[tid]['parents']:
                    self.tasks[p]['deps'] -= 1
                    if self.tasks[p]['deps'] == 0:
                        p_h = self.tasks[p]['height']
                        heapq.heappush(self.ready_queue, (-p_h, p))
                        
        return waves

# --- Run Configuration ---
NUM_TREES = 1000
LEAVES = 1000
BATCH_SIZE = 1024

sim = UnrootedTreeTaskSim(NUM_TREES, LEAVES)
sim.build_dependencies()
sim.compute_heights_and_init_queue()
waves = sim.run_simulation(BATCH_SIZE)

print(f"\n--- RESULTS for {NUM_TREES} Trees ---")
print(f"Total Waves: {len(waves)}")
print(f"First 20 Waves: {waves[:20]}")
print(f"Last 20 Waves:  {waves[-20:]}")
print(f"All Waves: {waves}")