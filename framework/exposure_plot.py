import matplotlib.pyplot as plt

def plot():
    gamma = 0.5
    def f(rel):
        return 0.8 * rel

    def exposure(i, rel=1):
        dropout = 1
        for j in range(i - 1):
            dropout *= (1 - f(rel))

        return gamma ** (i - 1) * (dropout)

    plt.plot( [1], [1], 'o', color='#ff00ff')
    plt.plot( [i for i in range(2,10)], [exposure(i) for i in range(2, 10)], 'ro')
    plt.plot( [i for i in range(2,10)], [exposure(i, rel=0) for i in range(2, 10)], 'bo')
    plt.plot( [i for i in range(1,10)], [exposure(i) for i in range(1, 10)], 'r')
    plt.plot( [i for i in range(1,10)], [exposure(i, rel=0) for i in range(1, 10)], 'b')
    plt.xlabel('Document rank')
    plt.ylabel('Exposure of each document')
    plt.title('Exposure for each document in ranking with λ=0.5')
    plt.legend(['Each document is irrelevant', 'Each document is relevant'])
    plt.show()

if __name__ == '__main__':
    plot()
