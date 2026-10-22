# MovieMate — Recommendation System (CodSoft AI Internship Task 4)

MovieMate recommends movies based on a movie you like or a free-text description of your preferences. It uses a content-based filtering approach with TF-IDF-style text vectors and cosine similarity.

## Features

- Recommend movies similar to a selected title
- Recommend based on text preferences, e.g. `space adventure`, `funny family`, or `mystery detective`
- Movie titles, genres, and descriptions included in a sample CSV dataset
- Fuzzy matching for small title typos
- Similarity scores for recommendations
- Interactive command-line menu
- Unit tests; no third-party dependencies

## Requirements

Python 3.9 or newer. The project uses only Python's standard library and works offline.

## Run

```bash
python recommender.py
```

Choose:
1. Recommend movies similar to a movie title.
2. Recommend movies based on your preferences.
3. Exit.

To customize the catalog, edit `movies.csv` and add rows with `title`, `genres`, and `description`.

## Example

For a title-based recommendation, enter `The Martian`. Movies with similar science-fiction and adventure content should rank highly.

For preference-based recommendations, enter `space adventure` or `comedy family`.

## Run tests

```bash
python -m unittest -v
```

## How it works

1. Combines each movie's genre and description into a text document.
2. Tokenizes text into lowercase words.
3. Builds TF-IDF-style weighted vectors. Repeated genre terms give genre information additional influence.
4. Uses cosine similarity to compare two vectors.
5. Ranks movies by similarity and excludes the input movie from title-based recommendations.

This is content-based filtering: recommendations are based on item descriptions, not other users' ratings. The sample catalog is intentionally small for a beginner-friendly internship project, so recommendations depend on the quality and variety of its descriptions.

## Project structure

```text
CODSOFT_TASK4/
├── recommender.py
├── movies.csv
├── test_recommender.py
├── README.md
└── .gitignore
```

## GitHub publishing

Follow the repository naming instructions from your internship coordinator. If one repository is intended for all tasks, place this project in a `Task4/` subfolder.

```bash
git init
git add .
git commit -m "Complete CodSoft Task 4: Movie Recommendation System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CODSOFT_TASKSNO.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name with your actual details.

## Suggested demo video (60–90 seconds)

1. Introduce the project as CodSoft AI Internship Task 4.
2. Show `movies.csv` and explain the metadata.
3. Recommend similar movies to `The Martian`.
4. Try preference-based recommendations using `mystery detective` or `comedy family`.
5. Briefly explain content-based filtering and cosine similarity.
6. Run `python -m unittest -v` and show your actual GitHub link.

## Suggested LinkedIn caption

> Completed Task 4 of my CodSoft Artificial Intelligence internship: a content-based Movie Recommendation System built with Python. It uses movie genres and descriptions to create TF-IDF-style vectors and cosine similarity scores, then recommends movies matching a title or text preferences. The project includes a sample dataset, interactive CLI, and unit tests.
>
> GitHub: [add your real repository link]
>
> @CODSOFT #codsoft #artificialintelligence #python #recommendationsystem #internship
