"""Content-based movie recommendation system for CodSoft Task 4."""

import csv
import re
from pathlib import Path
from difflib import get_close_matches
from math import sqrt


DEFAULT_DATASET = Path(__file__).with_name("movies.csv")


def tokenize(text):
    """Normalize text into simple word tokens without third-party libraries."""
    return re.findall(r"[a-z0-9]+", text.lower())


class MovieRecommender:
    """Recommend movies using content similarity and TF-IDF-style vectors."""

    def __init__(self, movies):
        self.movies = []
        for movie in movies:
            title = str(movie.get("title", "")).strip()
            genres = str(movie.get("genres", "")).strip()
            description = str(movie.get("description", "")).strip()
            if title:
                self.movies.append({
                    "title": title,
                    "genres": genres,
                    "description": description,
                    "text": f"{genres} {genres} {description}",
                })
        if not self.movies:
            raise ValueError("The movie dataset contains no valid movies.")
        self._build_vectors()

    @classmethod
    def from_csv(cls, csv_path=DEFAULT_DATASET):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Movie dataset not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return cls(list(csv.DictReader(file)))

    def _build_vectors(self):
        documents = [tokenize(movie["text"]) for movie in self.movies]
        vocabulary = sorted({token for doc in documents for token in doc})
        self.vocabulary = vocabulary
        document_frequency = {
            word: sum(1 for doc in documents if word in set(doc))
            for word in vocabulary
        }
        n_docs = len(documents)
        self.vectors = []
        for doc in documents:
            counts = {}
            for word in doc:
                counts[word] = counts.get(word, 0) + 1
            vector = {}
            for word, count in counts.items():
                # Smoothed inverse document frequency; repeated genre terms add weight.
                idf = 1.0 + (n_docs / (1 + document_frequency[word]))
                vector[word] = (1 + __import__("math").log(count)) * idf
            self.vectors.append(vector)

    @staticmethod
    def _cosine_similarity(left, right):
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(word, 0.0) for word, value in left.items())
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def find_movie_index(self, title):
        """Find exact title first, then a conservative close-title match."""
        target = title.strip().lower()
        for index, movie in enumerate(self.movies):
            if movie["title"].lower() == target:
                return index
        titles = [movie["title"] for movie in self.movies]
        matches = get_close_matches(title, titles, n=1, cutoff=0.72)
        if matches:
            return next(i for i, movie in enumerate(self.movies) if movie["title"] == matches[0])
        return None

    def recommend(self, title, limit=5):
        """Return similar movies sorted by cosine similarity, excluding the input movie."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        index = self.find_movie_index(title)
        if index is None:
            return []
        scores = []
        for other_index, other in enumerate(self.movies):
            if other_index == index:
                continue
            score = self._cosine_similarity(self.vectors[index], self.vectors[other_index])
            scores.append({
                "title": other["title"],
                "genres": other["genres"],
                "description": other["description"],
                "score": score,
            })
        scores.sort(key=lambda item: item["score"], reverse=True)
        return scores[:limit]

    def recommend_by_preferences(self, preferences, limit=5):
        """Recommend movies from free-text preferences such as 'space adventure funny'."""
        query = tokenize(preferences)
        if not query:
            return []
        query_doc = " ".join(query)
        query_vector = {}
        for word in query:
            if word not in self.vocabulary:
                continue
            tf = query.count(word)
            df = sum(1 for movie in self.movies if word in set(tokenize(movie["text"])))
            query_vector[word] = (1 + __import__("math").log(tf)) * (1.0 + len(self.movies) / (1 + df))
        if not query_vector:
            return []
        scored = []
        for movie, vector in zip(self.movies, self.vectors):
            score = self._cosine_similarity(query_vector, vector)
            scored.append({
                "title": movie["title"],
                "genres": movie["genres"],
                "description": movie["description"],
                "score": score,
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return [item for item in scored if item["score"] > 0][:limit]


def show_recommendations(recommendations):
    if not recommendations:
        print("No matching recommendations found. Try another title or different preferences.")
        return
    for rank, movie in enumerate(recommendations, start=1):
        print(f"{rank}. {movie['title']}  |  {movie['genres']}  |  similarity: {movie['score']:.3f}")
        print(f"   {movie['description']}")


def main():
    try:
        recommender = MovieRecommender.from_csv()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not load movie dataset: {exc}")
        return

    print("=" * 58)
    print("             MOVIE RECOMMENDATION SYSTEM")
    print("        CodSoft Artificial Intelligence - Task 4")
    print("=" * 58)
    print(f"Loaded {len(recommender.movies)} movies.")
    print("Choose: 1) Recommend similar movies  2) Recommend by preferences  3) Exit")

    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        if choice == "3":
            print("Thanks for using MovieMate. Enjoy your next movie!")
            break
        if choice == "1":
            title = input("Enter a movie you like: ").strip()
            count = input("How many recommendations? (default 5): ").strip()
            try:
                limit = int(count) if count else 5
                if limit < 1:
                    raise ValueError
            except ValueError:
                print("Please enter a positive whole number.")
                continue
            recommendations = recommender.recommend(title, limit)
            if not recommendations:
                print("Movie not found in the sample dataset. Try one of the listed titles in movies.csv.")
            else:
                print("\nMovies you may enjoy:")
                show_recommendations(recommendations)
        elif choice == "2":
            preferences = input("Describe what you like (e.g. space adventure funny): ").strip()
            count = input("How many recommendations? (default 5): ").strip()
            try:
                limit = int(count) if count else 5
                if limit < 1:
                    raise ValueError
            except ValueError:
                print("Please enter a positive whole number.")
                continue
            print("\nRecommendations based on your preferences:")
            show_recommendations(recommender.recommend_by_preferences(preferences, limit))
        else:
            print("Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
