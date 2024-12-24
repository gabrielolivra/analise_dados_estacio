import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Tratamento': ['Implante', 'Clareamento', 'Ortodontia', 'Restauração', 'Limpeza'],
    'Receita': [1500, 500, 2000, 300, 100],
    'Custo_Tratamento': [800, 200, 1200, 150, 50]
}

df = pd.DataFrame(data)


df['Margem_Lucro'] = df['Receita'] - df['Custo_Tratamento']
df['Margem_Lucro_Percentual'] = (df['Margem_Lucro'] / df['Receita']) * 100

print(df)


meses = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)  
receita = np.array([1200, 1350, 1250, 1450, 1550, 1600, 1700, 1800, 1900, 2000, 2100, 2200]) 


modelo = LinearRegression()


modelo.fit(meses, receita)

proximo_mes = np.array([[13]])
previsao = modelo.predict(proximo_mes)

plt.scatter(meses, receita, color='blue')  
plt.plot(meses, modelo.predict(meses), color='red') 
plt.scatter(proximo_mes, previsao, color='green') 
plt.title('Previsão de Receita Futura')
plt.xlabel('Mês')
plt.ylabel('Receita (R$)')
plt.show()

print(f'Previsão de receita para o próximo mês: R${previsao[0]:.2f}')
