#!/usr/bin/env python3
"""
Быстрый тест сбора данных (100 раундов)
"""
import sys
sys.path.insert(0, '/home/user/jjhbn/GGG3')

from collect_historical_data import connect_web3, collect_data
from web3 import Web3
import json

PREDICTION_ADDR = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"
PREDICTION_ABI = json.loads(r"""[
    {
        "inputs": [{"internalType": "uint256", "name": "epoch", "type": "uint256"}],
        "name": "rounds",
        "outputs": [
            {"internalType": "uint256", "name": "epoch", "type": "uint256"},
            {"internalType": "uint256", "name": "startTimestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "lockTimestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "closeTimestamp", "type": "uint256"},
            {"internalType": "int256", "name": "lockPrice", "type": "int256"},
            {"internalType": "int256", "name": "closePrice", "type": "int256"},
            {"internalType": "uint256", "name": "lockOracleId", "type": "uint256"},
            {"internalType": "uint256", "name": "closeOracleId", "type": "uint256"},
            {"internalType": "uint256", "name": "totalAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "bullAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "bearAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "rewardBaseCalAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "rewardAmount", "type": "uint256"},
            {"internalType": "bool", "name": "oracleCalled", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "currentEpoch",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")

def main():
    print("="*60)
    print("БЫСТРЫЙ ТЕСТ СБОРА ДАННЫХ (100 раундов)")
    print("="*60)

    try:
        # Подключаемся
        w3 = connect_web3()

        # Создаем контракт
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(PREDICTION_ADDR),
            abi=PREDICTION_ABI
        )

        # Собираем 100 раундов
        data = collect_data(
            w3,
            contract,
            n_rounds=100,
            output_file='test_100_rounds.json'
        )

        if data:
            print(f"\n✅ ТЕСТ ПРОЙДЕН! Собрано {len(data)} примеров")
            return True
        else:
            print(f"\n❌ ТЕСТ ПРОВАЛЕН!")
            return False

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
