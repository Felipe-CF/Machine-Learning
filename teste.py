import re
import json
import os
from pathlib import Path

def log_keras_para_json(arquivo_log):
    """
    Converte log do Keras (copiado do terminal) para JSON estruturado
    """
    try:
        with open(arquivo_log, 'r', encoding='utf-8') as f:
            texto = f.read()

        # Regex que captura nome da métrica e valor
        padrao = r'([a-zA-Z_]+):\s*([0-9.]+(?:e[+-]?[0-9]+)?)'

        # Extrair todas as métricas em ordem
        matches = re.findall(padrao, texto)

        # Construir dicionário
        metricas = {}
        for nome, valor_str in matches:
            if nome not in metricas:
                metricas[nome] = []
            try:
                valor = float(valor_str)
                metricas[nome].append(valor)
            except ValueError:
                metricas[nome].append(valor_str)

        return metricas

    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
        return None

def encontrar_melhor_auc_validacao_e_seu_treino(metricas):
    """
    Encontra o melhor AUC de validação e o AUC de treino no mesmo índice
    """
    auc_validacao = metricas.get('val_AUC', [])
    auc_treino = metricas.get('AUC', [])

    if not auc_validacao:
        return None, None

    # Encontrar o índice do melhor AUC de validação
    melhor_val_auc = max(auc_validacao)
    indice_melhor_val = auc_validacao.index(melhor_val_auc)

    # Pegar o AUC de treino no mesmo índice
    if indice_melhor_val < len(auc_treino):
        auc_treino_no_melhor_val = auc_treino[indice_melhor_val]
    else:
        auc_treino_no_melhor_val = None

    return melhor_val_auc, auc_treino_no_melhor_val

def gerar_nome_arquivo_json(arquivo_log, fold_numero, auc_treino_no_melhor_val, melhor_auc_validacao):
    """
    Gera o nome do arquivo JSON seguindo o padrão
    """
    # Extrair nome base do arquivo de log (sem extensão)
    nome_base = Path(arquivo_log).stem

    # Remover qualquer padrão existente de fold
    nome_limpo = re.sub(r'_kfold_\d+_', '_', nome_base)
    nome_limpo = re.sub(r'_fold_\d+_', '_', nome_limpo)
    nome_limpo = re.sub(r'_kfold_\d+$', '', nome_limpo)

    # Formatar os valores de AUC (8 casas decimais, substituindo ponto por underscore)
    auc_treino_str = f"{auc_treino_no_melhor_val:.8f}".replace('.', '_')
    auc_validacao_str = f"{melhor_auc_validacao:.8f}".replace('.', '_')

    # Construir nome do arquivo JSON
    nome_json = f"{nome_limpo}_kfold_{fold_numero}_fit_history_auc_{auc_treino_str}_val_auc_{auc_validacao_str}.json"

    return nome_json

def processar_log_keras(arquivo_log, fold_numero):
    """
    Processa o arquivo de log e salva JSON no mesmo diretório
    """
    # Verificar se arquivo existe
    if not os.path.exists(arquivo_log):
        print(f"Erro: Arquivo {arquivo_log} não encontrado!")
        return None

    # Extrair métricas
    metricas = log_keras_para_json(arquivo_log)

    if not metricas:
        print("Erro: Não foi possível extrair as métricas!")
        return None

    # Verificar se as métricas necessárias existem
    if 'val_AUC' not in metricas or 'AUC' not in metricas:
        print("Erro: Métricas 'AUC' ou 'val_AUC' não encontradas!")
        return None

    # Encontrar melhor AUC de validação e seu correspondente AUC de treino
    melhor_auc_validacao, auc_treino_correspondente = encontrar_melhor_auc_validacao_e_seu_treino(metricas)

    if melhor_auc_validacao is None or auc_treino_correspondente is None:
        print("Erro: Não foi possível determinar os valores de AUC!")
        return None

    # Gerar nome do arquivo JSON
    nome_json = gerar_nome_arquivo_json(arquivo_log, fold_numero, auc_treino_correspondente, melhor_auc_validacao)

    # Definir caminho completo (mesmo diretório do arquivo de log)
    diretorio_log = Path(arquivo_log).parent
    caminho_json = diretorio_log / nome_json

    # Salvar JSON
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2)

    print(f"JSON salvo: {caminho_json}")

    return caminho_json

if __name__ == "__main__":
    # Define o diretório base como o mesmo do script
    file_dir = os.path.dirname(os.path.abspath(__file__)) + '\\efficient_fit_history'

    # Define o caminho do arquivo de log
    file_path = os.path.join(file_dir, 'kfold2.txt')

    # Define o número do fold
    fold_numero = 2

    # Processa o arquivo
    processar_log_keras(file_path, fold_numero)