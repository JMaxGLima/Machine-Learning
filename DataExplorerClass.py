"""

Aula de Exploração de Dados

"""

import pandas as pd
from sklearn import datasets 
import random

class DataExplorer:

    def __init__(self, data):
        self.data = data
        self.iris = datasets.load_iris()
    

    @staticmethod
    def example_dataset():
        cities = ['New York', 'Paris', 'London', 'Tokyo', 'Beijing']
        countries = ['USA', 'France', 'UK', 'Japan', 'China']
        data = [
            * [{'name': f'Person {i}', 
                'age': round(random.uniform(18, 70)), 
                'city': f'{random.choice(cities)}', 
                'country': f'{random.choice(countries)}'} for i in range(1, 51)],
        ]
        return data

    # def age_country_correlation(self):     
    #     pass

    def show_iris(self):
        print(self.iris)

    def calc_corr_iris(self):
        df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        print(df.corr())

    def show_data(self):
        print(self.data)

def main():
    data = DataExplorer.example_dataset()
    explorer = DataExplorer(data)
    explorer.show_data()
    explorer.age_country_correlation()
    explorer.show_iris()
    explorer.calc_corr_iris()

if __name__ == "__main__":
    main()
