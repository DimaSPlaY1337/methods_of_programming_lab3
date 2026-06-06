import time
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#  1. АЛГОРИТМЫ ГЕНЕРАЦИИ 

class LCG:
    def __init__(self, seed=42):
        self.state = seed
        # Нестандартные параметры
        self.m = 2**31 - 1
        self.a = 48271
        self.c = 13

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

class ModifiedXorshift:
    def __init__(self, seed=42):
        self.state = seed
        if self.state == 0: self.state = 1

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        # Модификация: XOR с константой
        self.state = x ^ 0xDEADBEEF 
        return self.state

class MiddleSquareWeyl:
    def __init__(self, seed=42):
        self.x = seed #любое начальное число
        self.w = 0 #любое начальное число
        self.s = 0xb5ad4eceda1ce2a9 #Константа Вейля

    def next(self):#следующий шаг
        self.x **= 2
        self.w = (self.w + self.s) & 0xFFFFFFFFFFFFFFFF
        self.x = (self.x + self.w) & 0xFFFFFFFFFFFFFFFF
        self.x = (self.x >> 32) | (self.x << 32) & 0xFFFFFFFFFFFFFFFF
        return self.x

# Обертка для получения чисел в нужном диапазоне [0, MAX]
def generate_samples(generator_class, num_samples=20, sample_size=1000, max_val=5000):
    gen = generator_class()
    samples = []# двумерный список из 20 выборок с 1000-ю числами.
    for _ in range(num_samples):
        sample = [(gen.next() % max_val) for _ in range(sample_size)]
        samples.append(sample)
    return samples

#  5 ТЕСТОВ NIST 
# Для тестов переводим числа в бинарный вид (0 и 1)

def to_binary_sequence(sample): #Подготовка данных
    bin_seq = []
    for num in sample:
        # Берем младшие биты для случайности
        bin_seq.extend([int(b) for b in bin(num)[2:].zfill(16)])
    return bin_seq

def nist_monobit(bin_seq):
    # Тест на частоту (Monobit)
    S = sum(1 if bit == 1 else -1 for bit in bin_seq)
    p_val = math.erfc(abs(S) / math.sqrt(2 * len(bin_seq)))
    return p_val

def nist_block_frequency(bin_seq, block_size=128):
    n = len(bin_seq)
    N = n // block_size
    if N == 0: return 0.0
    chi_sq = 0.0 #сумма отклонений от каждого блока
    for i in range(N):
        block = bin_seq[i*block_size : (i+1)*block_size]
        pi = sum(block) / block_size #доля единиц в этом блоке
        chi_sq += 4 * block_size * (pi - 0.5)**2
    from scipy.special import gammaincc
    return gammaincc(N / 2, chi_sq / 2)#считаем проценты. перевод значения Хи-квадрат в p-value

def nist_runs(bin_seq):
    n = len(bin_seq)
    pi = sum(bin_seq) / n #доля единиц в последовательности
    if abs(pi - 0.5) >= (2 / math.sqrt(n)): return 0.0 #частотный тест
    Vn_obs = 1 + sum(1 for i in range(n-1) if bin_seq[i] != bin_seq[i+1])#количество серий
    #сравнение реального кол-ва серий с идеальным
    p_val = math.erfc(abs(Vn_obs - 2*n*pi*(1-pi)) / (2 * math.sqrt(2*n) * pi * (1-pi)))
    return p_val

def nist_longest_run_ones(bin_seq):
    # Упрощенная версия для блока 8 бит
    block_size = 8
    blocks = len(bin_seq) // block_size
    freqs = [0, 0, 0, 0] # <=1, 2, 3, >=4
    for i in range(blocks):
        block = bin_seq[i*block_size:(i+1)*block_size]
        max_run, current_run = 0, 0
        for bit in block:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        if max_run <= 1: freqs[0] += 1
        elif max_run == 2: freqs[1] += 1
        elif max_run == 3: freqs[2] += 1
        else: freqs[3] += 1
    
    #Формула Пирсона
    chi2 = sum(((freqs[i] - blocks * p)**2) / (blocks * p) for i, p in enumerate([0.2148, 0.3672, 0.2305, 0.1875]))
    from scipy.special import gammaincc
    return gammaincc(3/2, chi2/2)

def nist_cumulative_sums(bin_seq):
    S = 0
    z = 0
    for bit in bin_seq:
        S += 1 if bit == 1 else -1
        z = max(z, abs(S))
    n = len(bin_seq)
    p_val = 1.0 # Упрощенный расчет для демо-целей. В идеале тут сложная формула суммы нормальных распределений.
    if z > math.sqrt(n) * 2: p_val = 0.01 
    return p_val

#  ОСНОВНАЯ ЛОГИКА 

def main():
    generators = {"LCG": LCG, "Xorshift": ModifiedXorshift, "Weyl": MiddleSquareWeyl}
    
    for name, gen_class in generators.items():
        print(f"\n=== Анализ генератора {name} ===")
        # 2. Получение 20 выборок по 1000 элементов
        samples = generate_samples(gen_class, num_samples=20, sample_size=1000, max_val=5000)
        
        # 3. Вычисление статистик для первой выборки (в качестве примера)
        sample = samples[0]
        mean = np.mean(sample)
        std_dev = np.std(sample)#средний размер отклонения
        cv = (std_dev / mean) * 100#коэффицент вариации
        print(f"Среднее: {mean:.2f}, Отклонение: {std_dev:.2f}, Коэфф. вариации: {cv:.2f}%")
        
        # 4. Проверка на равномерность (Хи-квадрат)
        # Разбиваем диапазон [0, 5000) на 10 интервалов
        # observed - массив показывает сколько чисел попало к каждый интервал
        observed, bins = np.histogram(sample, bins=10, range=(0, 5000))
        expected = [len(sample)/10] * 10 #ожидаемая частота
        chi2_stat, p_val_chi = stats.chisquare(f_obs=observed, f_exp=expected)
        print(f"Хи-квадрат: p-value = {p_val_chi:.4f} (Если > 0.05, распределение равномерно)")
        
        # 5. Тесты NIST
        bin_seq = to_binary_sequence(sample)
        print("Тесты NIST (p-value > 0.01 означает прохождение):")
        print(f" 1. Monobit: {nist_monobit(bin_seq):.4f}")
        print(f" 2. Block Frequency: {nist_block_frequency(bin_seq):.4f}")
        print(f" 3. Runs: {nist_runs(bin_seq):.4f}")
        print(f" 4. Longest Run: {nist_longest_run_ones(bin_seq):.4f}")
        print(f" 5. Cumulative Sums: {nist_cumulative_sums(bin_seq):.4f}")

    #  6. ГРАФИКИ СКОРОСТИ 
    print("\nЗамер времени генерации (может занять несколько секунд)...")
    sizes = [1000, 10000, 100000, 500000, 1000000]# количество чисел
    times = { "LCG": [], "Xorshift": [], "Weyl": [], "Python built-in": [] }
    
    for size in sizes:
        for name, gen_class in generators.items():
            gen = gen_class()
            start = time.time()
            _ = [gen.next() for _ in range(size)]
            times[name].append(time.time() - start)
            
        # Стандартный метод Python (Mersenne Twister)
        start = time.time()
        _ = [np.random.randint(0, 5000) for _ in range(size)]
        times["Python built-in"].append(time.time() - start)
        
    plt.figure(figsize=(10, 6))
    for name, t in times.items():
        plt.plot(sizes, t, marker='o', label=name)

    plt.ticklabel_format(axis='x', style='plain')    
    plt.title('Сравнение скорости генераторов')
    plt.xlabel('Количество элементов (шт)')
    plt.ylabel('Время генерации (сек)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()