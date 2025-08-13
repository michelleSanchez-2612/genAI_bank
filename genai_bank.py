#pip install -U langchain-community
#pip install faiss-cpu
#pip install google-generativeai
#pip install sentence-transformers

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from langchain.embeddings import HuggingFaceEmbeddings
from dataclasses import dataclass
import torch
import faiss
import numpy as np
import requests
import time
import google.generativeai as genai
import os

############# RAG + FAISS ########################

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

class VectorStore:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents: List[Document] = []
        # Move embedding model to CPU
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
        self.index = None  #FAISS index
        self.embedding_dim = 384  # fixed dimension
        self.id_to_doc = {}

    def create_index(self, docs: List[Document]):
        #index the documents on FAISS index.
        self.documents = docs.copy()
        embeddings = []
        for i, doc in enumerate(self.documents):
            emb = self.embedder.embed_query(doc.text)
            emb_array = np.array(emb, dtype='float32')
            doc.embedding = emb_array
            embeddings.append(emb_array)
        embeddings_matrix = np.vstack(embeddings)

        faiss.normalize_L2(embeddings_matrix)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_matrix)
        self.id_to_doc = {i: doc for i, doc in enumerate(self.documents)}

    def query(self, text: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        #query the retrieved documents in FAISS index
        if self.index is None or len(self.documents) == 0:
            return []
        q_emb = self.embedder.embed_query(text)
        q_emb = np.array(q_emb, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            doc = self.id_to_doc[idx]
            results.append((doc, float(dist)))
        return results

############# LLM CLIENT ########################

class LLMClient:
    def __init__(self, model_name="gemini-1.5-flash"):
        genai.configure(api_key="AIzaSyC-eK1XxugXzScap-o-V8XmQzKrFeqWhwA") #test api key
        self.model = genai.GenerativeModel(model_name=model_name)

    def call(self, system_prompt: str, user_prompt: str, retrieved_docs, tool_results=None) -> str:
        # Combine prompts
        context = system_prompt + "\n"
        if retrieved_docs:
            context += "Knowledge base:\n"
            for doc, _ in retrieved_docs:
                context += f"- {doc.text}\n"
        if tool_results:
            context += "Bank tools:\n"
            for k, v in tool_results.items():
                context += f"- {k}: {v}\n"

        full_input = context + "\nUsuario: " + user_prompt + "\nAsistente:"

        try:
            response = self.model.generate_content(full_input)
            return response.text

        except Exception as e:
            print(f"Ocurrió un error al llamar a la API de Gemini: {e}")
            return "Lo siento, no pude generar una respuesta en este momento."

############# SIMULATED BANK TOOLS ########################

class BankTools:
    def __init__(self):
        # Example user in DB
        self.customers = {
            "user-123": {
                "name": "Ana Gomez",
                "accounts": {
                    "acc-001": {"type": "checking", "currency": "USD", "balance": 2540.75},
                    "acc-002": {"type": "savings", "currency": "USD", "balance": 10234.10},
                },
                "cards": {"card-abc": {"status": "active", "last4": "4242", "due_date": "2025-09-10"}},
                "investments": {"pol-01": {"type": "home", "status": "active", "expiry": "2026-04-30"}},
            }
        }

    def authenticate_session(self, session_token: str) -> Tuple[int, Dict[str, Any]]:
        if session_token == "valid-token":
            return 200, {"user_id": "user-123"}
        else:
            return 403, {"error": "Invalid token"}

    def get_account_summary(self, user_id: str, account_id: str):
        user = self.customers.get(user_id)
        if not user:
            return 404, {"error": "user not found"}
        acc = user["accounts"].get(account_id)
        if not acc:
            return 404, {"error": "account not found"}

        return 200, {"account_id": account_id, "type": acc["type"], "currency": acc["currency"], "balance": acc["balance"], "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    def get_recent_transactions(self, user_id: str, account_id: str, limit: int = 5):
        if user_id not in self.customers:
            return 404, {"error": "user not found"}
        if account_id not in self.customers[user_id]["accounts"]:
            return 404, {"error": "account not found"}
        txs = [
            {"id": "tx1", "date": "2025-08-10", "amount": -35.50, "description": "Coffee shop"},
            {"id": "tx2", "date": "2025-08-08", "amount": -120.00, "description": "Online shopping"},
            {"id": "tx3", "date": "2025-08-01", "amount": 1500.00, "description": "Salary"},
        ]
        return 200, {"transactions": txs[:limit]}

    def get_card_status(self, user_id: str, card_id: str):
        user = self.customers.get(user_id)
        if not user:
            return 404, {"error": "user not found"}
        card = user["cards"].get(card_id)
        if not card:
            return 404, {"error": "card not found"}
        return 200, {"card_id": card_id, "status": card["status"], "last4": card["last4"], "due_date": card["due_date"]}

    def get_investment_details(self, user_id: str, investment_id: str):
        user = self.customers.get(user_id)
        if not user:
            return 404, {"error": "user not found"}
        pol = user["investments"].get(investment_id)
        if not pol:
            return 404, {"error": "investment not found"}
        return 200, {"investment_id": investment_id, "type": pol["type"], "status": pol["status"], "expiry": pol["expiry"]}

############# ORCHESTRATOR ########################

class Orchestrator:
    def __init__(self, vector_store, llm_client, bank_tools, system_prompt):
        self.vs = vector_store
        self.llm = llm_client
        self.tools = bank_tools
        self.prompt = system_prompt
        self.tool_keywords = ["balance", "saldo", "transacciones", "movimientos", "tarjeta", "póliza", "inversión"]
        self.pending_request = None

    def execute_personal_request(self, incoming_text: str, user_id) -> str:
        lower = incoming_text.lower()
        account_id = "acc-001"
        card_id = "card-abc"
        investment_id = "pol-01"

        # Simple routing to Bank's API
        if "balance" in lower or "saldo" in lower:
            status, result = self.tools.get_account_summary(user_id, account_id)
            if status == 200:
                docs = self.vs.query(incoming_text, top_k=2)
                return self.llm.call(system_prompt=self.prompt, user_prompt=incoming_text, retrieved_docs=docs, tool_results={"account_summary": result})
            else:
                return f"Error retrieving account: {result.get('error', 'unknown error')}"

        elif "transacciones" in lower or "movimientos" in lower:
            status, result = self.tools.get_recent_transactions(user_id, account_id, limit=5)
            if status == 200:
                docs = self.vs.query(incoming_text, top_k=2)
                return self.llm.call(system_prompt=self.prompt, user_prompt=incoming_text, retrieved_docs=docs, tool_results={"transactions": result["transactions"]})
            else:
                return f"Error retrieving transactions: {result.get('error', 'unknown error')}"

        elif "tarjeta" in lower:
            status, result = self.tools.get_card_status(user_id, card_id)
            if status == 200:
                docs = self.vs.query(incoming_text, top_k=2)
                return self.llm.call(system_prompt=self.prompt, user_prompt=incoming_text, retrieved_docs=docs, tool_results={"card_status": result})
            else:
                return f"Error retrieving card: {result.get('error', 'unknown error')}"

        elif "inversión" in lower or "póliza" in lower:
            status, result = self.tools.get_investment_details(user_id, investment_id)
            if status == 200:
                docs = self.vs.query(incoming_text, top_k=2)
                return self.llm.call(system_prompt=self.prompt, user_prompt=incoming_text, retrieved_docs=docs, tool_results={"investment": result})
            else:
                return f"Error retrieving investment: {result.get('error', 'unknown error')}"

        else:
            return "No he podido entender tu requerimiento, ¿necesitas conocer sobre sus cuentas, transacciones, tarjetas o pólizas?"


    def handle_message(self, incoming_text: str, session_token: Optional[str] = None) -> str:
        lower = incoming_text.lower()
        wants_personal = any(k in lower for k in self.tool_keywords)

        if self.pending_request and session_token:
            user_id = self.tools.authenticate_session(session_token)[1]["user_id"]
            original_request = self.pending_request
            self.pending_request = None
            return self.execute_personal_request(original_request, user_id)

        # personal info
        if wants_personal:
            code, auth_payload = self.tools.authenticate_session(session_token or "")

            if code != 200:
                self.pending_request = incoming_text
                return "Por favor, debemos validar su identidad. Ingrese el código OTP enviado a su celular"

            user_id = auth_payload["user_id"]
            return self.execute_personal_request(incoming_text, user_id)

        # FAQ: retrieve RAG information and don't use Bank's API
        else:
            self.pending_request = None
            docs = self.vs.query(incoming_text, top_k=3)

            if not docs:
                # Agent without information
                derivation_prompt = "No se ha encontrado información relevante. Por favor, transfiera la consulta a un asesor."
                return self.llm.call(
                    system_prompt=self.prompt,
                    user_prompt=derivation_prompt,
                    retrieved_docs=None,
                    tool_results=None
                )
            else:
                # Agent found information
                return self.llm.call(
                    system_prompt=self.prompt,
                    user_prompt=incoming_text,
                    retrieved_docs=docs,
                    tool_results=None
            )

############# RAG DOCUMENTS INSTANTIATION ########################
rag_docs = [
    Document(id="doc-1", text="La Cuenta Corriente tiene una comisión mensual de mantenimiento de $10. Se exonera si el saldo promedio mensual supera los $2,500. El interés anual sobre ahorros es de 1.2%."),
    Document(id="doc-2", text="La multa por pago tardío en la tarjeta de crédito es de $30. El pago mínimo es el 3% del saldo del estado de cuenta. Para disputas, contacte al centro de soporte dentro de los 60 días."),
    Document(id="doc-3", text="Para abrir una cuenta corriente necesitas una identificación válida y comprobante de domicilio. La apertura en línea está disponible para clientes elegibles."),
    Document(id="doc-4", text="Nuestras políticas: la política de privacidad y términos establecen que los datos personales se procesan bajo consentimiento y se almacenan por no más de 5 años, salvo que la regulación indique lo contrario."),
    Document(id="doc-5", text="La Cuenta de Ahorros ofrece una tasa de interés anual del 1.5% y no tiene comisión por mantenimiento. Puedes retirar dinero en cualquier cajero automático sin costo adicional."),
    Document(id="doc-6", text="Ofrecemos diferentes tipos de cuentas: Cuenta Corriente, Cuenta de Ahorros, Cuenta para Jóvenes y Cuenta Empresarial, cada una con beneficios adaptados a tus necesidades."),
    Document(id="doc-7", text="El crédito inmediato es un préstamo personal preaprobado de hasta $5,000 dólares, con una tasa fija y plazos flexibles para su pago."),
    Document(id="doc-8", text="Para solicitar un crédito inmediato, el cliente debe contar con una cuenta activa y un historial crediticio positivo. La aprobación es rápida y sin papeleo adicional."),
    Document(id="doc-9", text="Las tasas de interés para créditos inmediatos comienzan en 12% anual, dependiendo del perfil crediticio del cliente."),
    Document(id="doc-10", text="La Cuenta para Jóvenes está diseñada para menores de 25 años, sin comisión de mantenimiento y con acceso a beneficios educativos y descuentos exclusivos."),
]
vs = VectorStore()
vs.create_index(rag_docs)


############# COMPONENTS INSTANTIATION ########################
system_prompt = """
Eres un asistente virtual del banco Guayaquil en Ecuador, encargado de responder preguntas frecuentes y consultas personalizadas de clientes.
Tus respuestas deben ser claras, concisas y respetar la privacidad del usuario.
Nunca reveles información personal a usuarios no autenticados.

Instrucciones para la respuesta:
1.  **Si se te proporciona un resultado de las herramientas del banco**, utiliza esa información para responder de forma directa, concisa y útil. No vuelvas a pedir autenticación.
2.  **Si la información es sobre detalles financieros (saldo, resumen de cuenta)**: Resume los datos clave de forma clara, indicando el monto y la moneda.
3.  **Si la información es sobre una lista de transacciones**: Presenta los movimientos más recientes de forma ordenada, mencionando la fecha, la descripción y el monto de cada uno.
4.  **Si la información es sobre un producto (póliza, tarjeta)**: Identifica los datos importantes como el estado (activo, inactivo), la fecha de vencimiento o el tipo de producto, y preséntalos de manera sencilla al usuario.
5.  **Si no tienes información de las APIs ni de la "Knowledge base"**, y la pregunta no es personal, informa al usuario que vas a transferir la consulta a un asesor humano.
6.  Si la pregunta del usuario es sobre una FAQ o información general, usa la "Knowledge base" proporcionada.
7.  Mantén las respuestas concisas y directas. No agregues detalles innecesarios.

"""

llm = LLMClient()
bank_tools = BankTools()
orch = Orchestrator(vector_store=vs, llm_client=llm, bank_tools=bank_tools, system_prompt=system_prompt)

############# TEST DEMO ########################

def run_chat_simulation(orchestrator):
    session_token = None
    last_user_input = None
    print("¡Bienvenido al asistente del Banco Guayaquil! Escribe 'salir' para terminar.\n")

    while True:
        user_input = input("Tú: ").strip()
        if user_input.lower() == "salir":
            print("Asistente: Gracias por usar el servicio. ¡Hasta luego!")
            break

        # Save user input
        last_user_input = user_input

        response = orchestrator.handle_message(user_input, session_token=session_token)
        print(f"Asistente: {response}")

        # Detect if OTP is needed
        if "Ingrese el código OTP" in response:
            otp_input = input("Ingrese OTP: ").strip()
            if otp_input == "123456":
                print("Asistente: OTP validado correctamente.")
                session_token = "valid-token"
                response_after_auth = orchestrator.handle_message(last_user_input, session_token=session_token)
                print(f"Asistente: {response_after_auth}")
            else:
                print("Asistente: OTP incorrecto. Por favor inténtalo de nuevo.")


run_chat_simulation(orch)