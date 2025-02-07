import numpy as np
from scipy import stats

class LinearRegressionAnalysis:
    def calculate_d(self, data):
        return data.shape[1] if data.ndim > 1 else 1

    def calculate_n(self, data):
        return data.shape[0] if data.ndim > 0 else 0

    def calculate_variance(self, data):
        return np.var(data, ddof=1) if data.ndim > 0 else 0

    def calculate_std_deviation(self, data):
        return np.std(data, ddof=1) if data.ndim > 0 else 0

    def report_regression_significance(self, x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }

    def calculate_r_squared(self, x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return r_value**2

    def calculate_ols_coefficients(self, X, Y):
        X_transpose = np.transpose(X)
        beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ Y
        return beta

    def check_linear_dependencies(self, data):
        num_features = data.shape[1]
        correlations = np.zeros((num_features, num_features))

        for i in range(num_features):
            for j in range(i + 1, num_features):
                r, _ = stats.pearsonr(data[:, i], data[:, j])
                correlations[i, j] = r
                correlations[j, i] = r  # Symmetrisk matris

        return correlations

    def check_pearson_correlation(self, data, feature1, feature2):
        r, _ = stats.pearsonr(data[:, feature1], data[:, feature2])
        return r

    def load_data_and_calculate(self, file_path):
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        x = data[:, 0]  # Byt ut med rätt kolumnindex för X
        y = data[:, 1]  # Byt ut med rätt kolumnindex för Y

        d = self.calculate_d(data)
        print(f'Antal funktioner (d): {d}')

        n = self.calculate_n(data)
        print(f'Storlek på datasetet (n): {n}')

        variance = self.calculate_variance(data)
        print(f'Varians: {variance}')

        std_deviation = self.calculate_std_deviation(data)
        print(f'Standardavvikelse: {std_deviation}')

        regression_significance = self.report_regression_significance(x, y)
        print(f'Regressionens signifikans: {regression_significance}')

        r_squared = self.calculate_r_squared(x, y)
        print(f'R^2-värde: {r_squared}')

        ols_coefficients = self.calculate_ols_coefficients(data[:, 0].reshape(-1, 1), data[:, 1])
        print(f'OLS-koefficienter: {ols_coefficients}')

        linear_dependencies = self.check_linear_dependencies(data)
        print(f'Linjära beroenden: {linear_dependencies}')

        pearson_correlation = self.check_pearson_correlation(data, 0, 1)
        print(f'Pearsons korrelation mellan variabel 1 och 2: {pearson_correlation}')

if __name__ == "__main__":
    analysis = LinearRegressionAnalysis()
    analysis.load_data_and_calculate("C:/Programmering/IT_hogskolan/Statistiska-metoder/Data/Small-diameter-flow.csv")
