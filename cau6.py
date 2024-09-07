from IPython.core.display import Latex
from networkx.algorithms.tournament import hamiltonian_path
from pandas.plotting import table
from sympy import symbols, true, false
from sympy.logic.boolalg import truth_table, to_cnf, And, Or, Not, Implies, Equivalent, BooleanTrue, BooleanFalse
from IPython.display import display, Math, Latex
import networkx as nx

# Cho G là đồ thị vô hướng
# Đồ thị có chu trình (đường đi) Euler không? Tại sao? Nếu có hãy chỉ ra một chu
# trình (đường đi) Euler của đồ thị.

import matplotlib.pyplot as plt

import networkx as nx
def is_hamiltonian_path(graph, path):
    """
    Kiểm tra xem đường đi hiện tại có phải là đường Hamilton không (không cần quay lại đỉnh đầu).
    """
    return len(path) == len(graph)

def find_hamiltonian_path(graph):
    """
    Tìm đường Hamilton trong đồ thị.
    """
    def backtrack(current_node, path):
        # Nếu tìm được đường đi Hamilton (đi qua tất cả các đỉnh một lần)
        if is_hamiltonian_path(graph, path):
            return path

        # Duyệt qua tất cả các đỉnh kề của đỉnh hiện tại
        for neighbor in graph.neighbors(current_node):
            if neighbor not in path:  # Chỉ tiếp tục nếu đỉnh chưa được thăm
                path.append(neighbor)
                result = backtrack(neighbor, path)  # Gọi đệ quy
                if result:
                    return result  # Nếu tìm thấy đường Hamilton, trả về kết quả
                path.pop()  # Backtrack nếu không tìm thấy

        return None

    # Bắt đầu tìm kiếm từ mỗi đỉnh
    for start_node in graph.nodes():
        path = [start_node]
        result = backtrack(start_node, path)
        if result:
            return result

    return None  # Nếu không có đường Hamilton

def find_hamiltonian_cycle(graph):
    for start_node in graph:
        path = [start_node]
        if is_hamiltonian_path(graph, path):
            return path + [start_node]  # Thêm đỉnh bắt đầu vào cuối chu trình để hoàn thành chu trình
    return None
# Tạo đồ thị với các cạnh và trọng số
G = nx.Graph()

def node_display(node, back_node,  weight, visited, last_node, start_node):
    # Hi
    if node == last_node:
        return f"{node} *".ljust(10)
    elif node in visited:
        return f"{node} -".ljust(10)
    return f"{node} ({weight}, {back_node or start_node})".ljust(10)

# Tìm đường đi ngắn nhất từ 1 đỉnh đến các đỉnh còn lại bằng Dijkstra ghi từng bước
def shortest_path(graph, start):
    s_paths = {start: [start]}
    visited = {start}
    nodes = set(graph.nodes())
    # Xác định trọng số của các đỉnh
    node_weights = {start: 0}
    node_backtrack = {start: None}
    sorted_nodes = sorted(nodes)
    for node in nodes:
        if node != start:
            node_weights[node] = float('inf')
            node_backtrack[node] = None

    while visited != nodes:
        min_path = None
        min_weight = float('inf')
        for node in visited:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    weight = graph[node][neighbor]['weight']
                    path = s_paths[node] + [neighbor]
                    if weight < min_weight:
                        min_weight = weight
                        min_path = path
                        last_point = neighbor
            # Cập nhật trọng số của các đỉnh
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    weight = graph[node][neighbor]['weight']
                    if node_weights[neighbor] > node_weights[node] + weight:
                        node_weights[neighbor] = node_weights[node] + weight
                        node_backtrack[neighbor] = node
        if last_point:
            # In ra bảng trọng số của các đỉnh và đỉnh trước đó
            print(", ".join([node_display(node, node_backtrack[node], node_weights[node], visited, last_point, start) for node in sorted_nodes]), f"Đỉnh đã xét {last_point}", f"Cạnh đã xét {min_path[-2]}{min_path[-1]}")

        if min_path:
            visited.add(min_path[-1])
            s_paths[min_path[-1]] = min_path
        else:
            break
    return s_paths

# tìm cây khung có trọng số nhỏ nhất
def find_mst(graph, start):
    mst = nx.Graph()
    visited = {start}
    nodes = set(graph.nodes())
    # Xác định trọng số của các đỉnh
    node_weights = {start: 0}
    node_backtrack = {start: None}
    sorted_nodes = sorted(nodes)
    for node in nodes:
        if node != start:
            node_weights[node] = float('inf')
            node_backtrack[node] = None

    while visited != nodes:
        min_path = None
        min_weight = float('inf')
        for node in visited:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    weight = graph[node][neighbor]['weight']
                    if weight < node_weights[neighbor]:
                        node_weights[neighbor] = weight
                        node_backtrack[neighbor] = node
                        min_path = (node, neighbor)
                        min_weight = weight
            # Cập nhật trọng số của các đỉnh
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    weight = graph[node][neighbor]['weight']
                    if node_weights[neighbor] > weight:
                        node_weights[neighbor] = weight
                        node_backtrack[neighbor] = node
        if min_path:
            visited.add(min_path[1])
            mst.add_edge(min_path[0], min_path[1], weight=min_weight)
        else:
            break
    return mst

edges = [
    'AB6', 'AD2', 'AE1', 'AF5',
    'BC10', 'BD5', 'BE2', 'BG3', 'BJ28',
    'CD12', 'CI1', 'CJ28',
    'DF3',
    'EF20', 'EG2',
    'FH30',
    'GH16', 'GJ1',
    'HI24', 'HJ4'
]

edges = [(edge[0], edge[1], int(edge[2:])) for edge in edges]

# Thêm các cạnh vào đồ thị
G.add_weighted_edges_from(edges)

# Kiểm tra xem đồ thị có chu trình Euler hay không
if nx.is_eulerian(G):
    eulerian_cycle = list(nx.eulerian_circuit(G))
    print("Chu trình Euler:", eulerian_cycle[0][0],"".join([f"{next_node} " for node, next_node in eulerian_cycle]))
# Kiểm tra xem đồ thị có đường đi Euler hay không
elif nx.is_semieulerian(G):
    eulerian_path = list(nx.eulerian_path(G))
    print("Đường đi Euler:", eulerian_path[0][0],"".join([f"{next_node} " for node, next_node in eulerian_path]))
else:
    print("Đồ thị không chứa chu trình hoặc đường đi Euler.")

# Kiểm tra chu trình hoặc đường Hamilton
hamiltonian_cycle = find_hamiltonian_cycle(G)
hamiltonian_path = find_hamiltonian_path(G)
if hamiltonian_cycle:
    print("Chu trình Hamilton:", " ".join(hamiltonian_cycle))
elif hamiltonian_path:
    print("Đường đi Hamilton:", " ".join(hamiltonian_path))
else:
    print("Đồ thị không chứa chu trình hoặc đường đi Hamilton.")

# Đường đi ngắn nhất từ F
shortest_paths = shortest_path(G, 'F')
for node, path in shortest_paths.items():
    print(f"Đường đi ngắn nhất từ F đến {node}: {' -> '.join(path)}")

# Tìm cây khung nhỏ nhất
print("Cây khung nhỏ nhất:")
mst = nx.minimum_spanning_tree(G)
path_in_mst = ",".join([f"{node}{neighbor}" for node, neighbor in mst.edges()])
sorted_path =  sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
for path in sorted_path:
    # kiểm tra path có trong cây khung nhỏ nhất không
    print(f"{path[2]['weight']}".ljust(10), f"{path[0]}{path[1]}".ljust(10), "Chọn" if f"{path[0]}{path[1]}" in path_in_mst else "Không chọn")
print("Cây khung nhỏ nhất:", mst.edges(data=True))

# Tìm cây khung lớn nhất
print("Cây khung lớn nhất:")
mst = nx.maximum_spanning_tree(G)
path_in_mst = ",".join([f"{node}{neighbor}" for node, neighbor in mst.edges()])
sorted_path =  sorted(G.edges(data=True), key=lambda x: -x[2]['weight'])
for path in sorted_path:
    # kiểm tra path có trong cây khung nhỏ nhất không
    print(f"{path[2]['weight']}".ljust(10), f"{path[0]}{path[1]}".ljust(10), "Chọn" if f"{path[0]}{path[1]}" in path_in_mst else "Không chọn")
print("Cây khung lớn nhất:", mst)
