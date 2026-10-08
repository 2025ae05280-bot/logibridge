data:
	python training/generate_dataset.py

train: data
	python training/train_model.py

variants: train
	python training/convert_ptq.py
	python training/prune_quantise.py

bench: variants
	python optimisation/benchmark.py

test:
	pytest -q

up:
	docker compose up --build

deploy:
	ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml

drift-demo:
	python -m monitoring.drift_monitor monitor --truck-id T01
