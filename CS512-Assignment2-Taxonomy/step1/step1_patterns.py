"""
I provide some utility code, you can choose to use them or not, as long as you can generate the correct "hypernymys.txt" file.
"""

import argparse
from collections import Counter, defaultdict

from pattern_matcher.hearstPatterns import HearstPatterns
from tqdm import tqdm

# from flashtext import KeywordProcessor


def read_sentences(file_path):
    sentences = []
    sentID2sentence = {}
    with open(file_path, "r") as fin:
        for lid, line in tqdm(enumerate(fin)):
            if lid == 0:  # skip header
                continue
            line = line.strip()
            if line:
                segs = line.split("\t")
                sentID = segs[0]
                sentence = segs[1]
                sentences.append([sentID, sentence])
                sentID2sentence[sentID] = sentence
    return sentences, sentID2sentence


def read_vocab(file_path):
    entityID2entityPreferredName = {}
    entityID2AllNames = {}
    with open(file_path, "r") as fin:    
        for lid, line in enumerate(fin):
            if lid == 0:  # skip header
                continue
            line = line.strip()
            if line:
                segs = line.split("\t")
                entityID = segs[0]
                entityPreferredName = segs[1]
                entityAllNames = [" ".join(cname.split("_")) for cname in segs[2].split("||")]
                entityID2entityPreferredName[entityID] = entityPreferredName
                entityID2AllNames[entityID] = entityAllNames
    return entityID2entityPreferredName, entityID2AllNames


def save_results(file_path, hypernyms, entityID2entityPreferredName):
    """
    suppose the variable "hypernyms" is of the below format: 
    {
        (parentID-1, childID-1): {sentID-a: num_occurrence_in_sentID-a, sentID-b: num_occurrence_in_sentID-b, ...},
        (parentID-2, childID-2): {sentID-c: num_occurrence_in_sentID-c, sentID-d: num_occurrence_in_sentID-d, ...},
        ...
    }
    """
    with open(file_path, "w") as fout:
        fout.write("HypernymID\tHypernym\tHyponymID\tHyponym\tEvidentalSentIDs\n")
        for ele in sorted(hypernyms.items(), key=lambda x:-len(x[1])):  # sort hypernym relations based on the number of their evidental sentencs
            parentID = ele[0][0]
            parentName = entityID2entityPreferredName[parentID]
            childID = ele[0][1]
            childName = entityID2entityPreferredName[childID]
            EvidentalSentIDs = ",".join([k+":"+str(v) for k,v in ele[1].items()])
            fout.write(f"{parentID}\t{parentName}\t{childID}\t{childName}\t{EvidentalSentIDs}\n")


def createUpdatedFile(entityID2entityPreferredName, entityID2AllNames, sentID2sentence ):
    file_Path = "../raw_data/updated_sentences.txt" 
    count = 0
    entityDic = {}
    preferredName2entityID = {}

    for entitiID in entityID2entityPreferredName:
         preferredName2entityID[entityID2entityPreferredName[entitiID]] = entitiID

    count = 0
    for entiID in entityID2AllNames:

        allEntities = entityID2AllNames[entiID]
        prefEntity = entityID2entityPreferredName[entiID]
        for entiti in allEntities:
            if(entiti[0]!= ' '):
                entiti = " " + entiti
            if(entiti[-1]!= ' '):
                entiti = entiti + " "
            entityDic[entiti] =  " NP_" + prefEntity + " "

    print("Dic created")

    with open(file_Path, "w") as fout:
        for sentId in sentID2sentence:
            count = count +1 
            sent = sentID2sentence[sentId].lower()
            output = " " + (sent + '.')[:-1]
            enTList = []
            for entiti in entityDic:                
                if entiti in sent:
                    enTList.append(entiti)
                 
            enTList.sort(key=len, reverse = True)

            for entiti in enTList:
                if entiti in output:
                    output = output.replace(entiti, entityDic[entiti])

            fout.write(f"{sentId}\t{output.strip()}\n")

            if(count%5000 == 0):
                print(count)
            


def main(args):
    sentences, sentID2sentence = read_sentences(args.corpus)
    entityID2entityPreferredName, entityID2AllNames = read_vocab(args.vocab)
    # createUpdatedFile(entityID2entityPreferredName, entityID2AllNames, sentID2sentence)

    hp = HearstPatterns(spacy_model_name = "", extended = True)
    count = 0
    hypernyms = {}
    preferredName2entityID = {}

    for entitiID in entityID2entityPreferredName:
        preferredName2entityID[entityID2entityPreferredName[entitiID]] = entitiID

    for lineId in sentID2sentence:
        line = sentID2sentence[lineId]
        pairs = hp.find_hyponyms(line, needs_chunk = False)
        if(len(pairs) > 0):
            for pair in pairs:
                pairKey  = ( preferredName2entityID[pair[1].replace(" ", "_")],  preferredName2entityID[pair[0].replace(" ", "_")])
                if pairKey in hypernyms:
                    lineDic = hypernyms[pairKey]
                    if lineId in lineDic:
                        lineDic[lineId] = lineDic[lineId] + 1
                    else:
                        lineDic[lineId] = 1
                else:
                    lineDic = {}
                    lineDic[lineId] = 1
                    hypernyms[pairKey] = lineDic


    print("We are here")
    save_results(args.output, hypernyms, entityID2entityPreferredName)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypernym Extraction by Pattern Matching')
    parser.add_argument('-corpus', default="../raw_data/updated_sentences.txt")
    parser.add_argument('-vocab', default="../raw_data/vocab.txt")
    parser.add_argument('-output', default="./hypernymys.txt")
    args = parser.parse_args()
    main(args)
