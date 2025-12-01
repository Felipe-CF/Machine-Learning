

dados = [
    {"c": "BR", "s": "ativo", "P": 120.50},
    {"c": "US", "s": "inativo", "P": 98.75},
    {"c": "FR", "s": "ativo", "P": 150.00},
    {"c": "JP", "s": "pendente", "P": 200.30},
    {"c": "DE", "s": "ativo", "P": 110.90}
]

s = 0

for d in dados:
    s+= d['p']

m = s/len(dados)

s = 0

for d in dados:
    if d['p'] > m:
        s+=1

print(s)