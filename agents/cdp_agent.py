import os
import json
from dotenv import load_dotenv
from typing_extensions import List

from cdp import Wallet
from cdp_langchain.utils import CdpAgentkitWrapper

load_dotenv()

abi_file_path = "./contracts/GranTEE.json"

abi=None
if os.path.exists(abi_file_path):
    with open(abi_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        abi=data["abi"]
    
def setup_wallet(wallet_data_file):
    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}
        
    agentkit = CdpAgentkitWrapper(**values)

    agentkit.wallet.save_seed_to_file("seed.txt")
    
    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    wallet_data = json.loads(wallet_data)

    wallet=Wallet.fetch(wallet_data["wallet_id"])
    wallet.load_seed_from_file("seed.txt")

    return wallet

wallet = setup_wallet("wallet_data.txt")

contract_address=os.environ.get("CONTRACT_ADDRESS") or "0x8bB9c2cf58DBbFcf4436035c332426c6cd49d3bA"

# example args: scholarshipId="10", applicant_address="0x....", new_status: "1"
def update_application_status(scholarshipId: str, applicant_address: str, new_status: str):
    invocation = wallet.invoke_contract(
    contract_address=contract_address,
    abi=abi,
    method="updateApplicationStatus",
    args={"_scholarshipId": scholarshipId, "_student": applicant_address, "_newStatus": new_status})
    
    invocation.wait()
    
# example args: scholarshipId="10", applicant_address="0x....", amount: "1000"
def send_scholarship(scholarshipId: str, applicant_address: str, amount: str):
    invocation = wallet.invoke_contract(
    contract_address=contract_address,
    abi=abi,
    method="sendScholarship",
    args={"_scholarshipId": scholarshipId, "_student": applicant_address, "_amount": amount})
    
    invocation.wait()
    

if __name__ == "__main__":
    update_application_status("1","0xD0A6F0F54803E50F27A6CC1741031094267AEE78", "1")
    send_scholarship("1","0xD0A6F0F54803E50F27A6CC1741031094267AEE78", "1000")
    
    
    
    