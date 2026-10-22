"""Unit tests for the content-based movie recommender."""

import unittest
from recommender import MovieRecommender


SAMPLE = [
    {"title": "Space Quest", "genres": "Sci-Fi Adventure", "description": "A space crew explores distant planets."},
    {"title": "Galaxy Mission", "genres": "Sci-Fi Adventure", "description": "Astronauts travel across the galaxy."},
    {"title": "Laughing Day", "genres": "Comedy", "description": "Friends share jokes in a small town."},
    {"title": "Ocean Mystery", "genres": "Mystery Thriller", "description": "A detective investigates a strange ocean event."},
]


class TestMovieRecommender(unittest.TestCase):
    def setUp(self):
        self.recommender = MovieRecommender(SAMPLE)

    def test_exact_title_finds_recommendations(self):
        results = self.recommender.recommend("Space Quest", 2)
        self.assertEqual(len(results), 2)
        self.assertNotIn("Space Quest", [item["title"] for item in results])

    def test_similar_movie_ranks_first(self):
        results = self.recommender.recommend("Space Quest", 3)
        self.assertEqual(results[0]["title"], "Galaxy Mission")

    def test_fuzzy_title_match(self):
        self.assertEqual(self.recommender.find_movie_index("Space Qest"), 0)

    def test_unknown_title_returns_empty(self):
        self.assertEqual(self.recommender.recommend("Unknown Film"), [])

    def test_preferences_return_matching_movies(self):
        results = self.recommender.recommend_by_preferences("space adventure", 3)
        self.assertTrue(results)
        self.assertEqual(results[0]["title"], "Space Quest")

    def test_empty_preferences_return_empty(self):
        self.assertEqual(self.recommender.recommend_by_preferences("   "), [])

    def test_invalid_limit_raises(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend("Space Quest", 0)


if __name__ == "__main__":
    unittest.main()
