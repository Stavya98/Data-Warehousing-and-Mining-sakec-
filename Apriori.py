from itertools import combinations

# Function to get frequent itemsets based on minimum support
def get_frequent_itemsets(transactions, min_support):
    itemsets = {}
    for transaction in transactions:
        for item in transaction:
            if item in itemsets:
                itemsets[item] += 1
            else:
                itemsets[item] = 1

    # Filter itemsets to only include those that meet or exceed the minimum support
    frequent_itemsets = {item: support for item, support in itemsets.items() if support >= min_support}
    return frequent_itemsets

# Function to generate candidate itemsets of size k
def get_candidate_itemsets(frequent_itemsets, k):
    candidates = []
    frequent_items = list(frequent_itemsets.keys())
    for combination in combinations(frequent_items, k):
        candidates.append(combination)
    return candidates


# Apriori algorithm to find all frequent itemsets
def apriori(transactions, min_support):
    k = 1
    # Initial set of frequent itemsets
    frequent_itemsets = get_frequent_itemsets(transactions, min_support)
    all_frequent_itemsets = [frequent_itemsets]

    # Iterate to find larger itemsets
    while frequent_itemsets:
        k += 1
        # Generate candidate itemsets of size k
        candidates = get_candidate_itemsets(frequent_itemsets, k)
        candidate_supports = {candidate: 0 for candidate in candidates}

        # Calculate support for each candidate itemset
        for transaction in transactions:
            for candidate in candidates:
                if set(candidate).issubset(set(transaction)):
                    candidate_supports[candidate] += 1

        # Filter candidate itemsets to only include those that meet or exceed the minimum support
        frequent_itemsets = {itemset: support for itemset, support in candidate_supports.items() if support >= min_support}
        if frequent_itemsets:
            all_frequent_itemsets.append(frequent_itemsets)

    return all_frequent_itemsets


transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter']
]

min_support = 2
frequent_itemsets = apriori(transactions, min_support)
print(frequent_itemsets)
