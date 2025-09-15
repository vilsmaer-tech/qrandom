import unittest
import random
import secrets
import numpy as np
import os
from scipy.stats import chisquare

# --- ПРАВИЛЬНЫЙ ИМПОРТ: разделяем по файлам ---
# Генераторы из random_01.py
from random_01 import (
    prng_flip,
    trng_flip
)

# Генераторы и статистика из random_02.py
from random_02 import (
    numpy_prng_flip,
    os_urandom_flip,
    calculate_stats,
    calculate_autocorrelation,
    calculate_histogram
)

# --- Фикстура: предсказуемый seed для воспроизводимости ---
class TestRNGFunctions(unittest.TestCase):

    def setUp(self):
        """Выполняется перед каждым тестом"""
        random.seed(42)
        np.random.seed(42)

    def test_prng_flip_returns_list_of_ints(self):
        """PRNG: возвращает список из n элементов 0 или 1"""
        n = 100
        result = prng_flip(n)
        self.assertEqual(len(result), n)
        self.assertTrue(all(x in {0, 1} for x in result))
        self.assertIsInstance(result, list)

    def test_trng_flip_returns_list_of_ints(self):
        """TRNG (secrets): возвращает список из n элементов 0 или 1"""
        n = 100
        result = trng_flip(n)
        self.assertEqual(len(result), n)
        self.assertTrue(all(x in {0, 1} for x in result))
        self.assertIsInstance(result, list)

    def test_numpy_prng_flip_returns_list_of_ints(self):
        """NumPy PRNG: возвращает список из n элементов 0 или 1"""
        n = 100
        result = numpy_prng_flip(n)  # ← теперь импортировано из random_02.py
        self.assertEqual(len(result), n)
        self.assertTrue(all(x in {0, 1} for x in result))
        self.assertIsInstance(result, list)

    def test_os_urandom_flip_returns_list_of_ints(self):
        """OS URANDOM: возвращает список из n элементов 0 или 1"""
        n = 100
        result = os_urandom_flip(n)  # ← теперь импортировано из random_02.py
        self.assertEqual(len(result), n)
        self.assertTrue(all(x in {0, 1} for x in result))
        self.assertIsInstance(result, list)

    def test_calculate_stats_all_zeros(self):
        """Проверка stats на массиве из одних нулей"""
        flips = [0] * 100
        stats = calculate_stats(flips)
        self.assertAlmostEqual(stats['mean'], 0.0, places=6)
        self.assertAlmostEqual(stats['deviation'], 0.5, places=6)
        self.assertAlmostEqual(stats['std'], 0.0, places=6)
        chi2_stat, p_val = chisquare([100, 0])
        self.assertAlmostEqual(stats['chi2_stat'], chi2_stat, places=6)
        self.assertLess(stats['p_value'], 1e-10)

    def test_calculate_stats_all_ones(self):
        """Проверка stats на массиве из одних единиц"""
        flips = [1] * 100
        stats = calculate_stats(flips)
        self.assertAlmostEqual(stats['mean'], 1.0, places=6)
        self.assertAlmostEqual(stats['deviation'], 0.5, places=6)
        self.assertAlmostEqual(stats['std'], 0.0, places=6)
        chi2_stat, p_val = chisquare([0, 100])
        self.assertAlmostEqual(stats['chi2_stat'], chi2_stat, places=6)
        self.assertLess(stats['p_value'], 1e-10)

    def test_calculate_stats_perfect_50_50(self):
        """Проверка stats на идеальном распределении 50/50"""
        flips = [0] * 50 + [1] * 50
        stats = calculate_stats(flips)
        self.assertAlmostEqual(stats['mean'], 0.5, places=6)
        self.assertAlmostEqual(stats['deviation'], 0.0, places=6)
        self.assertAlmostEqual(stats['std'], 0.5, places=6)
        self.assertGreaterEqual(stats['p_value'], 0.05)
        self.assertLess(stats['chi2_stat'], 3.84)

    def test_calculate_histogram(self):
        """Проверка гистограммы: возвращает [P(0), P(1)]"""
        flips = [0, 1, 0, 1, 0]
        hist = calculate_histogram(flips)
        self.assertEqual(len(hist), 2)
        self.assertAlmostEqual(hist[0], 0.6, places=6)
        self.assertAlmostEqual(hist[1], 0.4, places=6)
        self.assertAlmostEqual(sum(hist), 1.0, places=6)

    def test_calculate_autocorrelation_short_sequence(self):
        """Автокорреляция на короткой последовательности (меньше max_lag)"""
        flips = [0, 1]
        autocorr_vals, pvals = calculate_autocorrelation(flips, max_lag=10)
        self.assertEqual(len(autocorr_vals), 10)
        self.assertEqual(len(pvals), 10)
        self.assertTrue(all(v == 0.0 for v in autocorr_vals))
        self.assertTrue(all(p == 1.0 for p in pvals))

    def test_calculate_autocorrelation_perfect_alternating(self):
        """Автокорреляция на чередующейся последовательности: [0,1,0,1,...]"""
        flips = [i % 2 for i in range(100)]
        autocorr_vals, pvals = calculate_autocorrelation(flips, max_lag=5)

        self.assertAlmostEqual(autocorr_vals[0], -1.0, delta=0.05)
        self.assertAlmostEqual(autocorr_vals[1], 1.0, delta=0.05)
        self.assertAlmostEqual(autocorr_vals[2], -1.0, delta=0.05)
        self.assertTrue(all(p < 0.05 for p in pvals[:3]))

    def test_calculate_autocorrelation_random_sequence(self):
        """Автокорреляция на случайной последовательности — должна быть близка к 0"""
        random.seed(123)
        flips = [random.randint(0, 1) for _ in range(1000)]
        autocorr_vals, pvals = calculate_autocorrelation(flips, max_lag=10)

        avg_abs_corr = np.mean(np.abs(autocorr_vals))
        self.assertLess(avg_abs_corr, 0.1)
        significant_count = sum(p < 0.05 for p in pvals)
        self.assertLessEqual(significant_count, 2)

    def test_all_generators_produce_same_length(self):
        """Все генераторы выдают правильную длину при n=50"""
        n = 50
        generators = [prng_flip, trng_flip, numpy_prng_flip, os_urandom_flip]
        for gen in generators:
            result = gen(n)
            self.assertEqual(len(result), n)

    def test_trng_flip_uses_secrets(self):
        """Проверка, что trng_flip использует secrets.token_bytes"""
        result1 = trng_flip(10)
        result2 = trng_flip(10)
        self.assertNotEqual(result1, result2, "Два вызова trng_flip дали одинаковые результаты — подозрительно!")

    def test_calculate_stats_raises_on_empty_input(self):
        """Проверка обработки пустого списка"""
        with self.assertRaises(ValueError):
            calculate_stats([])


if __name__ == '__main__':
    print("🧪 Запуск unit-тестов для RNG сравнения...")
    unittest.main(verbosity=2)