import os
import sys
import psycopg2
import pandas as pd
from psycopg2 import sql
from sqlalchemy import create_engine, text
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging
import subprocess
from sqlalchemy.engine.url import URL
import pickle
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Adicione o diretório raiz do projeto ao PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.configuracao import obter_config_bd

logger = logging.getLogger(__name__)

class GerenciadorBD:
    def __init__(self):
        url = URL.create(
            drivername="postgresql",
            username=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME')
        )
        self.engine = create_engine(url, connect_args={"connect_timeout": 5})
        self.Session = sessionmaker(bind=self.engine)
        logger.debug("Engine do banco de dados criado")
        self.engine = create_engine(url, poolclass=QueuePool, pool_size=5, max_overflow=10)

    def testar_conexao(self):
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Conexão com o banco de dados bem-sucedida!")
                return result.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Erro ao conectar ao banco de dados: {str(e)}")
            raise
    
    def limpar_cache_antigo(self, dias=30):
        query = """
        DELETE FROM tweets_cache
        WHERE ultima_atualizacao < :data_limite
        """
        data_limite = datetime.now() - timedelta(days=dias)
        self.executar_query(query, {'data_limite': data_limite})
        logger.info(f"Cache mais antigo que {dias} dias foi limpo.")


    def conectar(self):
        try:
            return self.engine.connect()
        except Exception as e:
            logger.error(f"Erro ao conectar ao banco de dados: {str(e)}")
            raise

    def executar_transacao(self, queries):
        logger.debug(f"Iniciando transação com {len(queries)} queries")
        try:
            with self.conectar() as conn:
                with conn.begin():
                    for query in queries:
                        conn.execute(query)
            logger.debug("Transação concluída com sucesso")
        except Exception as e:
            logger.error(f"Erro na transação: {str(e)}", exc_info=True)
            raise

    

    def criar_tabela_modelos(self):
        query = """
        CREATE TABLE IF NOT EXISTS modelos_treinados (
            id SERIAL PRIMARY KEY,
            nome VARCHAR(100) UNIQUE,
            modelo_json TEXT,
            pesos BYTEA,
            data_criacao TIMESTAMP
        )
        """
        self.executar_query(query)

    def criar_tabelas(self):
        with self.engine.connect() as conn:
            # Tabela acoes
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS acoes (
                    id SERIAL PRIMARY KEY,
                    acao VARCHAR(10),
                    data DATE,
                    abertura FLOAT,
                    maxima FLOAT,
                    minima FLOAT,
                    fechamento FLOAT,
                    volume FLOAT,
                    CONSTRAINT acoes_unique UNIQUE (acao, data)
                )
            """))

            # Tabela acoes_normalizados
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS acoes_normalizados (
                    id SERIAL PRIMARY KEY,
                    acao VARCHAR(10),
                    data DATE,
                    abertura FLOAT,
                    maxima FLOAT,
                    minima FLOAT,
                    fechamento FLOAT,
                    volume FLOAT,
                    UNIQUE (acao, data)
                )
            """))

            # Tabela dados_economicos
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dados_economicos (
                    id SERIAL PRIMARY KEY,
                    indicador VARCHAR(50),
                    data DATE,
                    valor FLOAT,
                    UNIQUE (indicador, data)
                )
            """))

            # Tabela dados_economicos_normalizados
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dados_economicos_normalizados (
                    id SERIAL PRIMARY KEY,
                    indicador VARCHAR(50),
                    data DATE,
                    valor FLOAT,
                    UNIQUE (indicador, data)
                )
            """))

            # Tabela fiis
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fiis (
                    id SERIAL PRIMARY KEY,
                    fii VARCHAR(10),
                    data DATE,
                    abertura FLOAT,
                    maxima FLOAT,
                    minima FLOAT,
                    fechamento FLOAT,
                    volume FLOAT,
                    dividendos FLOAT,
                    UNIQUE (fii, data)
                )
            """))

            # Tabela fiis_normalizados
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fiis_normalizados (
                    id SERIAL PRIMARY KEY,
                    fii VARCHAR(10),
                    data DATE,
                    abertura FLOAT,
                    maxima FLOAT,
                    minima FLOAT,
                    fechamento FLOAT,
                    volume FLOAT,
                    dividendos FLOAT,
                    UNIQUE (fii, data)
                )
            """))

            # Tabela sentimento_mercado
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentimento_mercado (
                    id SERIAL PRIMARY KEY,
                    data DATE NOT NULL,
                    texto TEXT,
                    ticker VARCHAR(20) NOT NULL,
                    sentimento FLOAT NOT NULL,
                    CONSTRAINT sentimento_mercado_unique UNIQUE (data, ticker)
                )
            """))

            # Tabela modelos_treinados
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS modelos_treinados (
                    id SERIAL PRIMARY KEY,
                    nome VARCHAR(100) UNIQUE,
                    modelo_json TEXT,
                    pesos BYTEA,
                    data_criacao TIMESTAMP,
                    caminho_pesos VARCHAR(255)
                )
            """))

            # Tabela sentimento_mercado_normalizados
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentimento_mercado_normalizados (
                    id SERIAL PRIMARY KEY,
                    data DATE NOT NULL,
                    texto TEXT,
                    ticker VARCHAR(20) NOT NULL,
                    sentimento FLOAT NOT NULL,
                    CONSTRAINT sentimento_mercado_normalizados_unique UNIQUE (data, ticker)
                )
            """))

            conn.commit()

        logger.info("Todas as tabelas foram criadas ou já existem.")
        
    def criar_tabela_sentimento(self):
        self.criar_tabelas()  # Isso criará todas as tabelas, incluindo sentimento_mercado
        logger.info("Tabela sentimento_mercado criada ou já existente")

    def get_cached_tweets(self, ticker, dias):
        query = """
        SELECT data, texto
        FROM tweets_cache
        WHERE ticker = :ticker AND data >= :data_limite
        ORDER BY data DESC
        """
        data_limite = datetime.now() - timedelta(days=dias)
        params = {'ticker': ticker, 'data_limite': data_limite}
        return self.executar_query(query, params)

    def get_fallback_sentiments(self, ticker, dias):
        query = """
        SELECT data, sentimento
        FROM sentimento_mercado
        WHERE ticker = :ticker AND data >= :data_limite
        ORDER BY data DESC
        """
        data_limite = datetime.now() - timedelta(days=dias)
        params = {'ticker': ticker, 'data_limite': data_limite}
        return self.executar_query(query, params)
   
    def inserir_dados(self, tabela, dados):
        if not dados:
            logger.warning(f"Não há dados para inserir na tabela {tabela}")
            return

        try:
            dados_df = pd.DataFrame(dados)
            dados_df.to_sql(tabela, self.engine, if_exists='append', index=False)
            logger.info(f"Dados inseridos com sucesso na tabela {tabela}")
            
            # Verificar o conteúdo da tabela após a inserção
            verificacao = self.executar_query(f"SELECT * FROM {tabela} LIMIT 5")
            logger.debug(f"Primeiras 5 linhas da tabela {tabela} após inserção:\n{verificacao}")
        except Exception as e:
            logger.error(f"Erro ao inserir dados na tabela {tabela}: {str(e)}", exc_info=True)
            raise

    def executar_query(self, query, params=None):
        logger.debug(f"Executando query: {query}")
        try:
            with self.engine.connect() as conn:
                if isinstance(params, list):
                    # Executar inserção em lote
                    conn.execute(text(query), params)
                    conn.commit()
                    return None
                else:
                    result = conn.execute(text(query), params)
                    conn.commit()
                    if result.returns_rows:
                        return pd.DataFrame(result.fetchall(), columns=result.keys())
                    else:
                        return None
        except Exception as e:
            logger.error(f"Erro ao executar query: {str(e)}", exc_info=True)
            raise

    def atualizar_dados(self, tabela, condicao, novos_valores):
        try:
            set_clause = ', '.join([f"{k} = :{k}" for k in novos_valores.keys()])
            query = f"UPDATE {tabela} SET {set_clause} WHERE {condicao}"
            params = {**novos_valores, **dict([c.split('=') for c in condicao.split(' AND ')])}
            
            with self.engine.connect() as conn:
                conn.execute(text(query), params)
                conn.commit()
            logger.info(f"Dados atualizados com sucesso na tabela {tabela}")
        except Exception as e:
            logger.error(f"Erro ao atualizar dados na tabela {tabela}: {str(e)}", exc_info=True)
            raise

    def excluir_dados(self, tabela, condicao):
        comando = sql.SQL("DELETE FROM {} WHERE {}").format(
            sql.Identifier(tabela),
            sql.SQL(condicao)
        )
        
        with self.conectar() as conn:
            with conn.begin():
                conn.execute(comando)

    def obter_dados(self, tabela):
        try:
            comando = f"SELECT * FROM {tabela}"
            return self.executar_query(comando)
        except sqlalchemy.exc.ProgrammingError as e:
            if "não existe a relação" in str(e):
                logger.warning(f"A tabela {tabela} não existe. Criando...")
                self.criar_tabelas()  # Corrigido para chamar o método correto
                return pd.DataFrame()  # Retorna um DataFrame vazio
            else:
                raise

    def obter_indicadores_economicos(self):
        query = "SELECT DISTINCT indicador FROM dados_economicos"
        resultado = self.executar_query(query)
        return resultado['indicador'].tolist()

    def verificar_dados_suficientes(self, tabela):
        dados = self.gerenciador_bd.obter_dados(f"{tabela}_normalizados")
        if dados.empty:
            logger.warning(f"Não há dados na tabela {tabela}_normalizados")
            return False
        return len(dados) >= 15  

    def fazer_backup(self, caminho_backup):
        logger.info(f"Iniciando backup do banco de dados para {caminho_backup}")
        try:
            comando = f"pg_dump -h {self.config['host']} -U {self.config['user']} -d {self.config['database']} -f {caminho_backup}"
            subprocess.run(comando, shell=True, check=True)
            logger.info("Backup concluído com sucesso")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro ao fazer backup: {str(e)}", exc_info=True)
            raise

    def restaurar_backup(self, caminho_backup):
        logger.info(f"Iniciando restauração do banco de dados a partir de {caminho_backup}")
        try:
            comando = f"psql -h {self.config['host']} -U {self.config['user']} -d {self.config['database']} -f {caminho_backup}"
            subprocess.run(comando, shell=True, check=True)
            logger.info("Restauração concluída com sucesso")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro ao restaurar backup: {str(e)}", exc_info=True)
            raise

    def executar_migracao(self, caminho_script):
        logger.info(f"Executando script de migração: {caminho_script}")
        try:
            with open(caminho_script, 'r') as script:
                comandos = script.read()
            self.executar_transacao([comandos])
            logger.info("Migração concluída com sucesso")
        except Exception as e:
            logger.error(f"Erro ao executar migração: {str(e)}", exc_info=True)
            raise

    def verificar_tabela(self, tabela):
        try:
            query = text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{tabela}')")
            with self.engine.connect() as conn:
                result = conn.execute(query)
                exists = result.scalar()
            logger.info(f"Tabela {tabela} {'existe' if exists else 'não existe'}")
            return exists
        except Exception as e:
            logger.error(f"Erro ao verificar tabela {tabela}: {str(e)}", exc_info=True)
            return False
        
    def obter_dados_economicos(self, data_inicial, data_final):
        query = """
        SELECT data, indicador, valor
        FROM dados_economicos_normalizados
        WHERE data BETWEEN :data_inicial AND :data_final
        ORDER BY data
        """
        params = {'data_inicial': data_inicial, 'data_final': data_final}
        dados = self.executar_query(query, params)
        
        if dados.empty:
            logger.warning("Nenhum dado econômico encontrado para o período especificado")
            return pd.DataFrame()

        # Pivotear os dados para ter uma coluna para cada indicador
        dados_pivot = dados.pivot(index='data', columns='indicador', values='valor')
        dados_pivot.reset_index(inplace=True)
        
        return dados_pivot
        
    def salvar_modelo(self, nome_modelo, modelo_json, pesos):
        try:
            # Verificar se o diretório existe, se não, criar
            caminho_diretorio = "./modelos/"
            if not os.path.exists(caminho_diretorio):
                os.makedirs(caminho_diretorio)
                logger.info(f"Diretório {caminho_diretorio} criado.")

            # Salvar pesos do modelo em um arquivo
            caminho_pesos = os.path.join(caminho_diretorio, f"{nome_modelo}_pesos.pkl")
            with open(caminho_pesos, 'wb') as f:
                pickle.dump(pesos, f)

            query = """
            INSERT INTO modelos_treinados (nome, modelo_json, caminho_pesos, data_criacao)
            VALUES (:nome, :modelo_json, :caminho_pesos, CURRENT_TIMESTAMP)
            ON CONFLICT (nome) DO UPDATE
            SET modelo_json = :modelo_json, caminho_pesos = :caminho_pesos, data_criacao = CURRENT_TIMESTAMP
            """
            params = {'nome': nome_modelo, 'modelo_json': modelo_json, 'caminho_pesos': caminho_pesos}

            self.executar_query(query, params)
            logger.info(f"Modelo {nome_modelo} salvo com sucesso no banco de dados")
        except SQLAlchemyError as e:
            logger.error(f"Erro ao salvar modelo {nome_modelo}: {str(e)}", exc_info=True)
            # Tentativa de reconexão e nova tentativa de salvar o modelo
            try:
                self.engine.dispose()  # Fecha todas as conexões do pool
                self.engine = create_engine(self.engine.url, poolclass=QueuePool, pool_size=5, max_overflow=10)
                self.executar_query(query, params)
                logger.info(f"Modelo {nome_modelo} salvo com sucesso no banco de dados após reconexão")
            except SQLAlchemyError as e:
                logger.error(f"Erro ao salvar modelo {nome_modelo} após tentativa de reconexão: {str(e)}", exc_info=True)

    def carregar_modelo(self, nome_modelo):
        try:
            query = "SELECT modelo_json, caminho_pesos FROM modelos_treinados WHERE nome = :nome"
            resultado = self.executar_query(query, {'nome': nome_modelo})
            if not resultado.empty:
                modelo_json = resultado.iloc[0]['modelo_json']
                caminho_pesos = resultado.iloc[0]['caminho_pesos']
                with open(caminho_pesos, 'rb') as f:
                    pesos = pickle.load(f)
                return modelo_json, pesos
            return None, None
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {nome_modelo}: {str(e)}", exc_info=True)
            return None, None
    
    def upsert_dados(self, tabela, dados, chaves_unicas):
        if not dados:
            logger.warning(f"Não há dados para inserir/atualizar na tabela {tabela}")
            return

        try:
            colunas = list(dados[0].keys())
            placeholders = ', '.join([f':{col}' for col in colunas])
            update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in colunas if col not in chaves_unicas])
            
            query = f"""
                INSERT INTO {tabela} ({', '.join(colunas)})
                VALUES ({placeholders})
                ON CONFLICT ({', '.join(chaves_unicas)}) DO UPDATE SET
                {update_set}
            """
            
            self.executar_query(query, dados)
            logger.info(f"Dados inseridos/atualizados com sucesso na tabela {tabela}")
        except Exception as e:
            logger.error(f"Erro ao inserir/atualizar dados na tabela {tabela}: {str(e)}")
            raise

    def criar_indices(self):
        indices = [
            ("acoes", "acao"),
            ("acoes", "data"),
            ("fiis", "fii"),
            ("fiis", "data"),
            ("dados_economicos", "indicador"),
            ("dados_economicos", "data"),
            ("sentimento_mercado", "ticker"),
            ("sentimento_mercado", "data")
        ]
        for tabela, coluna in indices:
            query = f"CREATE INDEX IF NOT EXISTS idx_{tabela}_{coluna} ON {tabela} ({coluna})"
            self.executar_query(query)

    def limpar_banco_dados(self):
        tabelas = ['acoes', 'acoes_normalizados', 'fiis', 'fiis_normalizados', 
                'dados_economicos', 'dados_economicos_normalizados', 
                'sentimento_mercado', 'sentimento_mercado_normalizados', 'modelos_treinados']
        
        try:
            with self.engine.connect() as conn:
                for tabela in tabelas:
                    conn.execute(text(f"TRUNCATE TABLE {tabela} RESTART IDENTITY CASCADE"))
                conn.commit()
            logger.info("Todas as tabelas foram limpas com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao limpar o banco de dados: {str(e)}")
            raise

    def obter_ultima_atualizacao(self, tabela):
        query = f"SELECT MAX(data) FROM {tabela}"
        resultado = self.executar_query(query)
        return resultado.iloc[0][0] if not resultado.empty else None

    def obter_ultima_data(self, tabela, identificador=None):
        if tabela == 'acoes':
            coluna_id, coluna_data = 'acao', 'data'
        elif tabela == 'fiis':
            coluna_id, coluna_data = 'fii', 'data'
        elif tabela == 'dados_economicos':
            coluna_id, coluna_data = 'indicador', 'data'
        elif tabela == 'sentimento_mercado':
            coluna_id, coluna_data = 'ticker', 'data'
        else:
            raise ValueError(f"Tabela não reconhecida: {tabela}")

        if identificador:
            query = f"SELECT MAX({coluna_data}) FROM {tabela} WHERE {coluna_id} = :identificador"
            params = {'identificador': identificador}
        else:
            query = f"SELECT MAX({coluna_data}) FROM {tabela}"
            params = {}

        resultado = self.executar_query(query, params)
        return resultado.iloc[0, 0] if not resultado.empty and pd.notnull(resultado.iloc[0, 0]) else None
    
    def adicionar_restricao_unicidade(self, tabela, colunas):
        constraint_name = f"{tabela}_{'_'.join(colunas)}_unique"
        query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints
                WHERE constraint_name = '{constraint_name}'
            ) THEN
                ALTER TABLE {tabela}
                ADD CONSTRAINT {constraint_name} UNIQUE ({', '.join(colunas)});
            END IF;
        END $$;
        """
        self.executar_query(query)
        logger.info(f"Restrição de unicidade adicionada ou já existente para {tabela}: {colunas}")

    def verificar_cache_tweets(self, ticker, data_limite):
        query = """
        SELECT COUNT(*) FROM tweets_cache
        WHERE ticker = :ticker AND data >= :data_limite AND ultima_atualizacao >= :cache_limite
        """
        cache_limite = datetime.now() - timedelta(hours=6)  # Cache válido por 6 horas
        result = self.executar_query(query, {'ticker': ticker, 'data_limite': data_limite, 'cache_limite': cache_limite})
        return result.iloc[0, 0] > 0

    def obter_tweets_do_cache(self, ticker):
        query = """
        SELECT data, texto FROM tweets_cache
        WHERE ticker = :ticker
        ORDER BY data DESC
        """
        return self.executar_query(query, {'ticker': ticker})

    def atualizar_cache_tweets(self, ticker, df):
        # Primeiro, removemos os tweets antigos para este ticker
        delete_query = "DELETE FROM tweets_cache WHERE ticker = :ticker"
        self.executar_query(delete_query, {'ticker': ticker})



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    gerenciador = GerenciadorBD()
    gerenciador.criar_tabelas()
    gerenciador.criar_tabela_tweets_cache()