# CS512 Spring 2020 Assignment 2 : Taxonomy Construction and Expansion

Version 1.0. Released on March 29th, 2020. Due on May 4th, 2020, 11:59PM, CST

## Prerequisites and Requirements

1. This is an **individual** assignment. You can discuss this assignment in Piazza, including the performance that your implementation can achieve on the provided dataset, but please do not work together or share codes, which will be considered as violation of the rule of this assignment. 
2. Please use **Python3** as your programming language and **Linux** as your operating system. 
3. Late policy:
	- 10% off for one day (May 5th, 11:59PM, CST),
	- 20% off for two working days (May 6th, 11:59PM, CST),
	- 40% off for three working days (later than May 6th, 11:59PM, CST)).

## Introduction

Let's consider the following conversation:

```
Alice: How is your quarantine life?
Bob: Pretty good, I just purchased a new Alienware with a powerful RTX 2080 GPU.
Alice: Wow, that must be great for World of Warcraft or LoL.
Bob: Enn... I actually bought it for my data mining course project, but I guess it definitely improve the gaming experience. 
Alice: Oo, I almost forget you are a CS major student. Good luck with your project.
```

The above dialogue, though looks simple, actually requires many shared knowledge between two participants. For example, Alice understands **Alienware** is a type of **computer** and thus can be used to play **online games** such as **World of Warcraft** and **LoL**. Similarly, when Bob says he uses the new computer for **data mining** course project, Alice recalls that Bob is a CS major student because she knows **data mining** is a subarea of **computer science**. Those shared knowledge about general-specific term pairs (a.k.a. hypernym-hyponym relations) are organized in a taxonomy structure. 

In this mini-research assignment, you will first design and implement two common types of hypernymy discovery strategies. Then, you will explore how to organize those extracted hypernym-hyponym relations into a tree/DAG-structured taxonomy. Finally, you will research on how to incrementally update an existing taxonomy by inserting a set of new emerging terms. 

We provide a substantial amount of code substrates for the first two parts (i.e., hypernymy discovery and taxonomy organization), which reduces your actual coding workload to less than 100 lines of code (if you do everything right, of course). For the last part, you need to conduct more research and come up with our own taxonomy expansion method which could be a very simple model. Finally, you need to write a report summarizing your findings and thinkings. **You are free to use any external resources (e.g., Wikipedia) and open-sourced libraries** as long as you clearly document them in the report. 


## Step-by-step Instructions

### Step 0: Check Datasets (2pt)

There are three files in `raw_data` directory. 

1. sentences.txt: The first row is header. Starting from the second row, each line represents a sentence and has two columns (separated by "\t"). The first column is the sentID and the second column is the tokenized sentence. 
2. sentences.sample.txt: A subset of sentences.txt used to self-check your algorithms' output results in later steps. 
3. vocab.txt: The first row is header. Starting from the second row, each line represents an entity (synset) and has three columns (separated by "\t"). The first column is the entityID, the second column is the preferred (canonical) name of this entity, and the third column is all names of this entity (seperated by "||"). 

**Discussions in Report**: Please write the number of sentences in `sentences.txt` and the number of entity synsets in `vocab.txt`.

### Step 1: Hypernymy Discovery by Pattern Matching (18pt)

The first type of hypernymy discovery method is **pattern matching**. One of the earlies pattern matching work, named Hearst patterns [1], designs six lexical-syntactic patterns to identify hypernym-hyponym relations in documents. For example, the pattern "NPx such as NP1, NP2, ..." can match the sentence "There are many machine learning techniques such as logistic regression, support vector machine, and neural network.", which derives the hypernymy relations (machine learning techniques, logistic regression), (machine learning techniques, support vector machine), and (machine learning techniques, neural network). 

In this part, you need to complete `./step1/step1_patterns.py` which applies Hearst patterns on the given `sentences.txt`, extracts a set of hypernym-hyponym relations, and saves extracted pairs together with the sentences supporting those extractions into a file named `hypernymys.txt`. We provide you with a file named `hypernymys.sample.txt` which contains a _minimum set_ of pairs that can be extracted from the file `sentences.sample.txt`. You can test your algorithm on `sentences.sample.txt` and the output pairs should be a superset of relations in `hypernymys.sample.txt`. 

**Discussions in Report**: The pattern matching method is certainly not perfect. There are two types of errors: (1) _false positive_ in which an incorrect relation is extracted, and (2) _false negative_ in which a correct relation is missed. Can you identify one common pattern for _false positive_ error and one pattern for _false negative_ error? Discuss your findings and potential solutions to resolve them.

### Step 2: Hypernymy Discovery via Distribution Method (20pt)

The second type of hypernymy discovery method is **distribution method**. This type of method assumes the context set of a hypernym is a superset of the context set of a hyponym. Such an assumption is called 'distributional inclusion hypothesis' (DIH). In order to measure to what extent the DIH holds for a given term pair, people develop a set of DIH scoring functions, including weeds_precision [3], clarkeDE [4], and invCL [4]. You may find their more concise definition in paper [2]. 

In this part, you need to implement three functions in `./step2/distribution_methods/model.py` and test the model performance on 800 term pairs in `./step2/test_pairs.txt` using `eval_DIH.py`. For your reference, the weeds_prec method should achieve AP score at least 0.45 and AP@100 score at least 0.70. 

**Discussions in Report**: You need to include the evaluation results returned by `eval_DIH.py` in your report. Also, you need to compare those three scoring functions and discuss how to possibly improve them. 

**Bonus (5pt, in addition to the 20 full pts)**: The original DIH definition does not specify what is the **context set** of a term. We currently use the sentences where a term appear as its context set. However, you may come up with other definitions of context sets. Also, you may design your own DIH scoring functions. Finally, you may resort to term embeddings or an external knowledge base for hypernymy discovery. Explore more here and try to outperform the current best method (among weeds_prec, clarkeDE, and invCL) in terms of AP and AP@100. Discuss your findings in the report. 


### Step 3: Taxonomy Construction via Graph Induction (25pt)

Hypernmym discovery focues on predicting the relation between a given term pair. Taxonomy construction goes one step further and tries to organize a set of terms into a hierarchical structure. One common approach is to first construct a directed term graph where each node represents a term and each directed edge indicates a possible hypernym-hyponym relation, and then prunes this graph into a tree/DAG-structured taxonomy. There are many proposed graph pruning algorithms [5] among which two most popular ones are: (1) NoCyc: which iteratively finds one cycle in the term graph and breaks the cycle by removing one selected edge in the cycle, and (2) DMST: which finds a directed maximum spanning tree in the term graph.

Before proceed to this part, **please first finish step 2 and then copy the entire `./step2/distribution_methods` directory under the `./step3/` directory**. Then, you need to implement the NoCyc and DMST algorithms (the simple version without synonym grouping described in paper [5]) in `./step3/taxonomy_organization/graph2dag.py`. Finally, you need to run `./step3/eval_e2e.py` which constructs the entire taxonomy, compares it with the ground truth reference taxonomy in `./step3/cs_taxonomy.txt`, and outputs the results into `final_e2e_results.txt`. 

**Discussions in Report**: You need to include the evaluation results returned by `eval_e2e.py` in your report. Also, you need to compare the results returned by different combinations of DIH plus graph pruning algorithms, and discuss their relative pros and cons. 

**Bonus (5pt, in addition to the 25 full pts)**: Design and implement your own algorithm that achieves better end-to-end taxonomy construction performance than the methods I provided. Discuss your findings in the report. 


### Step 4: Taxonomy Expansion (15pt)

With the fast-growing volume of human knowledge, existing taxonomies will become outdated 
and fail to capture emerging concepts. As a result, we need to dynamically update a taxonomy to incorporate new concepts. 

In this part, you need to conduct more research and propose your own method that insert a set of new terms into the existing taxonomy. We provide two files: (1) the existing taxonomy is included in `./step3/cs_taxonomy.txt` where each line represents a taxonomy edge and includes two terms separated by "\t". The first term is the hypernym and the second term is the hyponym, and (2) the new concept list in `./step4/new_concepts_list.txt` where each line represents a new concept to be inserted into the existing taxonomy. For each new concept, you need to predict its possible parent node(s) as its insertion position(s). 

**Discussions in Report**: You need to clearly describe your taxonomy expansion method in your report and discuss its pros and cons. 


### Step 5: Report Writing (20pt)

The final report should be a pdf file and include all the elements in above "Discussions in Report" subsections. 

## What to submit? 

1. step1_patterns.py (in step 1)
2. hypernyms.txt (output in step 1)
3. model.py (in step 2)
4. graph2dag.py (in step 3)
5. final_e2e_results.py (output in step 3)
6. any model codes in step 4
7. a file including parent node(s) for each new concepts (output in step 4)
8. report**.pdf**

Put all the above files into one directory. Rename that directory to "CS512_Assignment2_Taxonomy_<NetID>" where <NetID> is your netid. Zip that directory and upload to Compass2g. 

## Frequently Asked Questions

I will constantly update this section later. 

## References

[1]. [Automatic Acquisition of Hyponyms from Large Text Corpora](http://people.ischool.berkeley.edu/~hearst/papers/coling92.pdf)

[2]. [Hearst Patterns Revisited: Automatic Hypernym Detection from Large Text Corpora](https://www.aclweb.org/anthology/P18-2057/)

[3]. [Characterising measures of lexical distributional similarity](https://www.aclweb.org/anthology/C04-1146/)

[4]. [Identifying hypernyms in distributional semantic spaces](https://www.aclweb.org/anthology/S12-1012/)

[5]. [Comparing Constraints for Taxonomic Organization](https://www.aclweb.org/anthology/N18-1030/)

[6]. [TaxoExpan: Self-supervised Taxonomy Expansion with Position-Enhanced Graph Neural Network](https://arxiv.org/abs/2001.09522)

[7]. [Expanding Taxonomies with Implicit Edge Semantics](http://emaadmanzoor.com/papers/20-www-arborist.pdf)
