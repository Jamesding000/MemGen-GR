

# =====================================================
# Task 1 -- Sequential Recommendation -- 17 Prompt
# =====================================================

seqrec_prompts = []

#####——0
prompt = {}
prompt["instruction"] = "The user has interacted with items {inters} in chronological order. Can you predict the next possible item that the user may expect?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "I find the user's historical interactive items: {inters}, and I want to know what next item the user needs. Can you help me decide?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Here are the user's historical interactions: {inters}, try to recommend another item to the user. Note that the historical interactions are arranged in chronological order."
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Based on the items that the user has interacted with: {inters}, can you determine what item would be recommended to him next?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "The user has interacted with the following items in order: {inters}. What else do you think the user need?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Here is the item interaction history of the user: {inters}, what to recommend to the user next?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = "Which item would the user be likely to interact with next after interacting with items {inters}?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "By analyzing the user's historical interactions with items {inters}, what is the next expected interaction item?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "After interacting with items {inters}, what is the next item that could be recommended for the user?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "Given the user's historical interactive items arranged in chronological order: {inters}, can you recommend a suitable item for the user?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Considering the user has interacted with items {inters}. What is the next recommendation for the user?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "What is the top recommended item for the user who has previously interacted with items {inters} in order?"
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——12
prompt = {}
prompt["instruction"] = "The user has interacted with the following items in the past in order: {inters}. Please predict the next item that the user most desires based on the given interaction records."
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = "Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. The historical interactions are provided as follows: {inters}."
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = "You can access the user's historical item interaction records: {inters}. Now your task is to recommend the next potential item to him, considering his past interactions."
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = "You have observed that the user has interacted with the following items: {inters}, please recommend a next item that you think would be suitable for the user."
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = "You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. Using this history as a reference, please select the next item to recommend to the user."
prompt["target"] = "{item}"
seqrec_prompts.append(prompt)



# ========================================================
# Task 2 -- Item2Index -- 19 Prompt
# ========================================================


item2index_prompts = []

# ========================================================
# Title2Index

#####——0
prompt = {}
prompt["instruction"] = "Which item has the title: \"{title}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "Which item is assigned the title: \"{title}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "An item is called \"{title}\", could you please let me know which item it is?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Which item is called \"{title}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "One of the items is named \"{title}\", can you tell me which item this is?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "What is the item that goes by the title \"{title}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)


# ========================================================
# Description2Index

#####——6
prompt = {}
prompt["instruction"] = "An item can be described as follows: \"{description}\". Which item is it describing?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "Can you tell me what item is described as \"{description}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "Can you provide the item that corresponds to the following description: \"{description}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)


#####——9
prompt = {}
prompt["instruction"] = "Which item has the following characteristics: \"{description}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Which item is characterized by the following description: \"{description}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "I am curious to know which item can be described as follows: \"{description}\". Can you tell me?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

# ========================================================
# Title and Description to index

#####——12
prompt = {}
prompt["instruction"] = "An item is called \"{title}\" and described as \"{description}\", can you tell me which item it is?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = "Could you please identify what item is called \"{title}\" and described as \"{description}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = "Which item is called \"{title}\" and has the characteristics described below: \"{description}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = "Please show me which item is named \"{title}\" and its corresponding description is: \"{description}\"."
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = "Determine which item this is by its title and description. The title is: \"{title}\", and the description is: \"{description}\"."
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——17
prompt = {}
prompt["instruction"] = "Based on the title: \"{title}\", and the description: \"{description}\", answer which item is this?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)

#####——18
prompt = {}
prompt["instruction"] = "Can you identify the item from the provided title: \"{title}\", and description: \"{description}\"?"
prompt["target"] = "{item}"
item2index_prompts.append(prompt)




# ========================================================
# Task 3 -- Index2Item --17 Prompt
# ========================================================

index2item_prompts = []

# ========================================================
# Index2Title

#####——0
prompt = {}
prompt["instruction"] = "What is the title of item {item}?"
prompt["target"] = "{title}"
index2item_prompts.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "What title is assigned to item {item}?"
prompt["target"] = "{title}"
index2item_prompts.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Could you please tell me what item {item} is called?"
prompt["target"] = "{title}"
index2item_prompts.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Can you provide the title of item {item}?"
prompt["target"] = "{title}"
index2item_prompts.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "What item {item} is referred to as?"
prompt["target"] = "{title}"
index2item_prompts.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Would you mind informing me about the title of item {item}?"
prompt["target"] = "{title}"
index2item_prompts.append(prompt)

# ========================================================
# Index2Description

#####——6
prompt = {}
prompt["instruction"] = "Please provide a description of item {item}."
prompt["target"] = "{description}"
index2item_prompts.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "Briefly describe item {item}."
prompt["target"] = "{description}"
index2item_prompts.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "Can you share with me the description corresponding to item {item}?"
prompt["target"] = "{description}"
index2item_prompts.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "What is the description of item {item}?"
prompt["target"] = "{description}"
index2item_prompts.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "How to describe the characteristics of item {item}?"
prompt["target"] = "{description}"
index2item_prompts.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "Could you please tell me what item {item} looks like?"
prompt["target"] = "{description}"
index2item_prompts.append(prompt)


# ========================================================
# Index to Title and Description

#####——12
prompt = {}
prompt["instruction"] = "What is the title and description of item {item}?"
prompt["target"] = "{title}\n\n{description}"
index2item_prompts.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = "Can you provide the corresponding title and description for item {item}?"
prompt["target"] = "{title}\n\n{description}"
index2item_prompts.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = "Please tell me what item {item} is called, along with a brief description of it."
prompt["target"] = "{title}\n\n{description}"
index2item_prompts.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = "Would you mind informing me about the title of the item {item} and how to describe its characteristics?"
prompt["target"] = "{title}\n\n{description}"
index2item_prompts.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = "I need to know the title and description of item {item}. Could you help me with that?"
prompt["target"] = "{title}\n\n{description}"
index2item_prompts.append(prompt)





