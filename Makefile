.PHONY: build run clean

BIN := ssm_go
ASOF ?= 2025-11-14

build:
	go build -o $(BIN) ./cmd/ssm

run: build
	SSM_ASOF_DATE=$(ASOF) ./$(BIN)

clean:
	rm -f $(BIN)
	rm -f outputs/*.csv
