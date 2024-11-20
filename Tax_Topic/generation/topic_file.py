import json
import regex
from utils import *
import argparse
import tqdm


# Function to read a JSONL file and process the "response" field
def topic_list_gen(
    topics_root,
    topics_list,
    lines,
    verbose,
    early_stop=100000,
):
    """
    Generate topics from documents using LLMs
    - topics_root, topics_list: Tree and list of topics generated from previous iteration
    - line: lines of the output files
    - verbose: Whether to print out results
    - early_stop: Threshold for topic drought (Modify if necessary)
    """
    topics = []
    topic_format = regex.compile("^\\[(\\d+)\\] ([\\w\\s]+):(.+)")
    running_dups = 0
    # Check if the "responses" key exists
    for line in lines:   
        topics = line.split("\n")
        for t in topics:
            t = t.strip()
            if regex.match(topic_format, t):
                groups = regex.match(topic_format, t)
                lvl, name, desc = (
                    int(groups[1]),
                    groups[2].strip(),
                    groups[3].strip(),
                )
                if lvl == 1:
                    dups = [s for s in topics_root.descendants if s.name == name]
                    if len(dups) > 0:  # Update count if topic already exists
                        dups[0].count += 1
                        running_dups += 1
                        if running_dups > early_stop:
                            return topics_list, topics_root
                    else:  # Add new topic if topic doesn't exist
                        new_node = Node(
                            name=name,
                            parent=topics_root,
                            lvl=lvl,
                            count=1,
                            desc=desc,
                        )
                        topics_list.append(f"[{new_node.lvl}] {new_node.name}")
                        running_dups = 0
                else:
                    if verbose:
                        print("Lower-level topics detected. Skipping...")
        if verbose:
            print(f"Topics: {line}")
            print("--------------------")
    return topics_list, topics_root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="top_topic_sample",
        help="the jsonl files to generate topics from",
    )
    parser.add_argument(
        "--seed_file",
        type=str,
        default="prompt/tax_seed1.md",
        help="markdown file to read the seed topics from",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/top_topic_2.md",
        help="file to write topics to",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="whether to print out results"
    )
    args = parser.parse_args()
    
    
    topics_root, topics_list = generate_tree(read_seed(args.seed_file))
    # Open the JSONL file and read line by line
    df = pd.read_json(str(args.output_file), lines=True)
    lines = df["responses"].tolist() 
    
    topics_list, topics_root= topic_list_gen(
        topics_root,
        topics_list,
        lines,
        args.verbose
    )
    
    with open(args.topic_file, "w") as f:
        print(tree_view(topics_root), file=f)
        
        
           
if __name__ == "__main__":
    main()
           
    