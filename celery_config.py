from celery import Celery

app = Celery('finaiti',
             broker='redis://localhost:6379/0',
             include=['app.tarefas'])

if __name__ == '__main__':
    app.start()