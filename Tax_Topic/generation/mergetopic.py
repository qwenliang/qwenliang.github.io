import pandas as pd
from utils import *
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import os


def topic_pairs(topic_sent, all_pairs, threshold=0.5, num_pair=2):
    """
    Return the most similar topic pairs and the pairs that have been prompted so far
    - topic_sent: List of topic sentences (topic label + description)
    - all_pairs: List of all topic pairs being prompted so far
    - threshold: Threshold for cosine similarity
    - num_pair: Number of pairs to return
    """
    # Calculate cosine similarity between all pairs of sentences
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sbert.encode(topic_sent, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    over, pairs, dups = [], [], []
    for i in range(len(cosine_scores)):
        for j in range(len(cosine_scores)):
            if i != j and sorted([i, j]) not in dups:
                pairs.append({"index": [i, j], "score": cosine_scores[i][j]})
                dups.append(sorted([i, j]))

    # Sort and choose num_pair pairs with scores higher than a certain threshold
    pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)
    count, idx = 0, 0
    while count < num_pair and idx < len(pairs):
        i, j = pairs[idx]["index"]
        if float(pairs[idx]["score"]) > threshold:
            if (sorted([topic_sent[i], topic_sent[j]]) not in all_pairs) and (
                topic_sent[i] != topic_sent[j]
            ):
                over.append([topic_sent[i], topic_sent[j]])
                all_pairs.append(sorted([topic_sent[i], topic_sent[j]]))
                count += 1
        idx += 1
    return [item for sublist in over for item in sublist], all_pairs


def merge_topics(
    topics_root,
    topics_node,
    prompt,
    deployment_name,
    temperature,
    max_tokens,
    top_p,
    verbose,
):
    """
    Prompt model to merge similar topics
    - topics_root: Root node of topic tree
    - topics_node: List of all nodes in topic tree
    - prompt: Prompt to be used for refinement
    - deployment_name: Model name
    - temperature: Temperature
    - max_tokens: Max tokens to generate
    - top_p: Top-p
    - verbose: Whether to print out results
    """
    # Get new pairs to be merged
    topic_sent = [
        f"[{topic.lvl}] {topic.name}: {topic.desc}" for topic in topics_root.descendants
    ]
    labels = [f"[{topic.lvl}] {topic.name}" for topic in topics_root.descendants]
    new_pairs, all_pairs = topic_pairs(
        topic_sent, all_pairs=[], threshold=0.5, num_pair=2
    )

    responses, removed, orig_new = [], [], {}
    # Pattern to match generations
    top_pattern = regex.compile(
        "^\[(\d+)\]([\w\s\-',]+)(?:[:\(\)\w\s\/])*?:([\w\s,\.\-\/;']+) \(((?:\[\d+\] [\w\s\-',]+(?:, )*)+)\)$"
    )
    # Pattern to match original topics being merged
    orig_pattern = regex.compile("(\[(?:\d+)\](?:[\w\s\-',]+)),?")

    while len(new_pairs) > 1:
        # Format topics to be merged in the prmpt
        inp, inp_label = [], []
        for topic in new_pairs:
            label = topic.split(":")[0]
            if label not in inp_label:
                inp.append(topic)
                inp_label.append(label)
        refiner_input = "\n".join(inp)
        refiner_prompt = prompt.format(Topics=refiner_input)
        if verbose:
            print(refiner_input)

        try:
            input_len = num_tokens_from_messages(refiner_input, "gpt-4-turbo")
            response = api_call(
                refiner_prompt, deployment_name, temperature, max_tokens, top_p
            )
            responses.append(response)
            merges = response.split("\n")
            for merge in merges:
                match = regex.match(regex.compile(top_pattern), merge.strip())
                if match:
                    lvl, name, desc = (
                        int(match.group(1)),
                        match.group(2).strip(),
                        match.group(3).strip(),
                    )
                    origs = [
                        t.strip(", ")
                        for t in regex.findall(orig_pattern, match.group(4).strip())
                    ]
                    orig_count = 0
                    add = False
                    if len(origs) > 1:
                        for node in topics_root.descendants:
                            if (
                                f"[{node.lvl}] {node.name}" in origs
                                and f"[{node.lvl}] {node.name}" != f"[{lvl}] {name}"
                            ):
                                orig_new[
                                    f"[{node.lvl}] {node.name}:"
                                ] = f"[{lvl}] {name}:"
                                if (
                                    f"[{node.lvl}] {node.name}: {node.desc}"
                                    in topic_sent
                                ):
                                    if verbose:
                                        print(
                                            f"Removing [{node.lvl}] {node.name}: {node.desc}\n"
                                        )
                                    topic_sent.remove(
                                        f"[{node.lvl}] {node.name}: {node.desc}"
                                    )
                                if f"[{node.lvl}] {node.name}" != f"[{lvl}] {name}":
                                    removed.append(f"[{node.lvl}] {node.name}")
                                orig_count += node.count
                                topics_node.remove(node)
                                node.parent = None
                                add = True
                        if add and f"[{lvl}] {name}" not in removed:
                            if (
                                f"[{lvl}] {name}: {desc}" not in topic_sent
                                and f"[{lvl}] {name}" not in labels
                            ):
                                new_node = Node(
                                    parent=topics_root,
                                    lvl=lvl,
                                    name=name,
                                    desc=desc,
                                    count=orig_count,
                                )
                                if verbose:
                                    print(f"Adding [{lvl}] {name}: {desc}\n")
                                topic_sent.append(f"[{lvl}] {name}: {desc}")
                                topics_node.append(new_node)
                            else:
                                if verbose:
                                    print(f"[{lvl}] {name} already exists!\n")
                                for node in topics_root.descendants:
                                    if f"[{node.lvl}] {node.name}" == f"[{lvl}] {name}":
                                        node.count += orig_count
        except:
            print("Error when calling API!")
            traceback.print_exc()
        print("--------------------")
        # Choose new pairs
        new_pairs, all_pairs = topic_pairs(
            topic_sent, all_pairs, threshold=0.5, num_pair=2
        )
    return responses, topics_root, orig_new


def remove_topics(topics_root, verbose, threshold=0.01):
    """
    Remove low-frequency topics from topic tree
    - topics_root: Root node of topic tree
    - verbose: Whether to print out results
    - Threshold: Percentage of all topic counts
    """
    topic_count = sum([node.count for node in topics_root.descendants])
    threshold = topic_count * threshold
    for node in topics_root.descendants:
        if node.count < threshold and node.lvl == 1:
            if verbose:
                print(f"Removing {node.name} ({node.count} counts)")
            node.parent = None
    return topics_root

