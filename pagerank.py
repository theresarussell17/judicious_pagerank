import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    ex: damping_factor = .85
        probability = 1 - damping_factor = .15

        corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
        page = "1.html"
        probability = 0.15
        dict = {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}

        This is because with probability 0.85, we choose randomly to go from page 1 to either page 2 or page 3 (so each of page 2 or page 3 has probability 0.425 to start), but every page gets an additional 0.05 because with probability 0.15 we choose randomly among all three of the pages
    """

    probList = {}

    # every page has equal probability of being chosen
    if not corpus[page]:
        for current_page in corpus:
            probList[current_page] = 1 / len(corpus)
    else:
        min_probability = (1 - damping_factor) / len(corpus)
        # set base probability
        for current_page in corpus:
            probList[current_page] = min_probability
        # overwrites base probability for only the pages that are linked to page
        for key in corpus[page]:
            probList[key] = min_probability + damping_factor / len(corpus[page])

    return probList


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample_count = 1
    pageRanks = {}
    for page in corpus:
        pageRanks[page] = 0

    # choose a random page to start
    pageChoice = random.choice(list(corpus.keys()))
    pageRanks[pageChoice] += 1

    while sample_count < n:
        probabilities = transition_model(corpus, pageChoice, damping_factor)
        pageChoice = random.choices(list(corpus.keys()), weights=list(probabilities.values()))[0]
        pageRanks[pageChoice] += 1
        sample_count += 1

    for page in pageRanks:
        pageRanks[page] /= n

    return pageRanks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRanks = {}
    for page in corpus:
        pageRanks[page] = 1/len(corpus)

    while True:
        newPageRanks = {}
        for page in corpus:
            newPageRanks[page] = (1 - damping_factor) / len(corpus)
            # which pages link to page
            for link in corpus:
                # check for when the page has no links first
                if not corpus[link]:
                    newPageRanks[page] += damping_factor * pageRanks[link] / len(corpus)
                elif page in corpus[link]:
                    newPageRanks[page] += damping_factor * pageRanks[link] / len(corpus[link])
        # sum the changes and check for convergence
        sumChange = 0
        for page in corpus:
            sumChange += abs(newPageRanks[page] - pageRanks[page])
        pageRanks = newPageRanks
        if sumChange < 0.001:
            break

    return pageRanks


if __name__ == "__main__":
    main()
