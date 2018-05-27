# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)

af = search.GPSProblem('A', 'F', search.romania)

print("Anchura:")
print search.breadth_first_graph_search(ab).path()
print("Profundidad:")
print search.depth_first_graph_search(ab).path()
print search.iterative_deepening_search(ab).path()
print search.depth_limited_search(ab).path()
print search.depth_limited_search(ab).path()
print("Branch_and_Bound ab")
print search.branch_and_bound_search(ab).path()
print("Branch_and_Bound_with_Subestimation ab")
print search.branch_and_bound_with_subestimation_search(ab).path()

print("Branch_and_Bound af")
print search.branch_and_bound_search(af).path()
print("Branch_and_Bound_with_Subestimation af")
print search.branch_and_bound_with_subestimation_search(af).path()



# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
