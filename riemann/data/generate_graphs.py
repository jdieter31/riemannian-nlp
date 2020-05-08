import argparse

def run(args):
    vertices = []
    edges = []
    if args.type == "cycle" or args.type == "treecycle":
        for i in range(args.length - 1):
            edges.append([i, i + 1])
        edges.append([args.length - 1, 0])
        vertices += range(args.length - 1)
    else:
        vertices = [0]

    if args.type == "treecycle":
        # Make a copy to avoid infinite loop
        for v in list(vertices):
            insert_tree(vertices, v, edges, args.depth, args.branching)
    
    if args.type == "tree":
        insert_tree(vertices, 0, edges, args.depth, args.branching)

    save_to_csv(args.out, edges)


def save_to_csv(file_name, edges):
    with open(file_name, "w+") as f:
        f.write("id1\tid2\tweight\n")
        for edge in edges:
            f.write(f"node{edge[0]}\tnode{edge[1]}\t1\n")

def insert_tree(vertices, root_vertex, edges, depth, branching):
    if depth == 0:
        return

    start_vertex = max(vertices) + 1
    vertices += range(start_vertex, start_vertex + branching)
    for i in range(branching):
        edges.append([root_vertex, start_vertex + i])
        insert_tree(vertices, start_vertex + i, edges, depth - 1, branching)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool to train graph embeddings \
                                 as detailed in "Retrofitting Manifolds to \
                                 Semantic Graphs"')
    parser.add_argument('-b', '--branching', type=int, default=3, help= \
        "Amount of branching for trees")
    parser.add_argument('-t', '--type', type=str, default="tree", help= \
        "Type of graph to generate options are cycle, tree, treecycle")
    parser.add_argument('-d', '--depth', type=int, default=3, help= \
                        "Depth of trees") 
    parser.add_argument('-l', '--length', type=int, default=10, help= \
                        "Length of cycle")
    parser.add_argument('-o', '--out', type=str)
    parser.set_defaults(func=run)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
