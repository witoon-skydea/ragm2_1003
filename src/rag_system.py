from typing import List, Dict, Any, Optional, Union, Tuple
from langchain.docstore.document import Document
import time
import os
from pathlib import Path
import logging
import shutil

# นำเข้าโมดูลต่างๆ ของระบบ
from src.document_processing.loader import DocumentLoader
from src.document_processing.chunking import EnhancedTextSplitter
from src.embeddings.embeddings_manager import ThaiCustomEmbeddings
from src.vector_store.vector_store import EnhancedVectorStore
from src.retrieval.retriever import EnhancedRetriever
from src.retrieval.hybrid_search import HybridSearchRetriever
from src.generation.llm_manager import OllamaClient, RAGPromptManager
from src.metrics.evaluation import RAGMetrics
from src.utils.helpers import (
    setup_directories, save_json, load_json, add_document_ids,
    format_doc_for_display, timer, create_timestamp
)

import config

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("thai_rag")

class ThaiRAGSystem:
    """ระบบ RAG ชั้นสูงสำหรับภาษาไทย ที่รวมทุกส่วนเข้าด้วยกัน"""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        use_hybrid_search: bool = True
    ):
        """
        เริ่มต้นระบบ RAG
        
        Args:
            data_dir: ไดเร็กทอรีสำหรับข้อมูล
            persist_dir: ไดเร็กทอรีสำหรับบันทึก vector store
            embedding_model: ชื่อโมเดล embeddings
            llm_model: ชื่อโมเดล LLM
            use_hybrid_search: ใช้ hybrid search หรือไม่
        """
        # ตั้งค่าไดเร็กทอรี
        self.data_dir = data_dir or str(config.DATA_DIR)
        self.persist_dir = persist_dir or str(config.PERSIST_DIR)
        
        # ตั้งค่าโมเดล
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL
        self.llm_model_name = llm_model or config.LLM_MODEL
        
        # ตั้งค่า hybrid search
        self.use_hybrid_search = use_hybrid_search
        
        # สร้างไดเร็กทอรีที่จำเป็น
        self.logs_dir = os.path.join(config.BASE_DIR, "logs")
        setup_directories([self.data_dir, self.persist_dir, self.logs_dir])
        
        # ตัวแปรสำหรับเก็บส่วนประกอบต่างๆ
        self.documents = []
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.hybrid_retriever = None
        self.ollama_client = None
        self.prompt_manager = None
        self.metrics = None
        
        # เริ่มต้นส่วนประกอบที่จำเป็น
        self._init_components()
        
        logger.info(f"เริ่มต้นระบบ RAG สำเร็จ โดยใช้โมเดล embeddings: {self.embedding_model_name}")
    
    def clear_db(self) -> None:
        """
        ล้างการเก็บข้อมูลและสร้าง vector store ใหม่
        """
        logger.info("กำลังล้างฐานข้อมูล vector store...")
        
        # ลบไดเร็กทอรี vector store เก่า
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            logger.info(f"ลบไดเร็กทอรี {self.persist_dir} แล้ว")
        
        # สร้างไดเร็กทอรีใหม่
        os.makedirs(self.persist_dir, exist_ok=True)
        logger.info(f"สร้างไดเร็กทอรี {self.persist_dir} ใหม่แล้ว")
        
        # ล้างรายการเอกสารที่โหลดไว้แล้ว
        self.documents = []
        
        # รีเซต vector store และ retriever
        self.vector_store = EnhancedVectorStore(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.retriever = None
        self.hybrid_retriever = None
        
        logger.info("ล้างฐานข้อมูล vector store เสร็จสมบูรณ์")
        
        print("ล้างฐานข้อมูล vector store เสร็จสมบูรณ์ พร้อมสำหรับการใช้งานใหม่")
    
    def _init_components(self) -> None:
        """เริ่มต้นส่วนประกอบต่างๆ ของระบบ"""
        # เริ่มต้น embeddings
        logger.info(f"กำลังเริ่มต้นโมเดล embeddings: {self.embedding_model_name}")
        self.embeddings = ThaiCustomEmbeddings(model_name=self.embedding_model_name)
        
        # เริ่มต้น vector store
        logger.info("กำลังเริ่มต้น vector store")
        self.vector_store = EnhancedVectorStore(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        # เริ่มต้น LLM client
        logger.info(f"กำลังเริ่มต้นโมเดล LLM: {self.llm_model_name}")
        self.ollama_client = OllamaClient(model_name=self.llm_model_name)
        
        # เริ่มต้น prompt manager
        logger.info("กำลังเริ่มต้น prompt manager")
        self.prompt_manager = RAGPromptManager(ollama_client=self.ollama_client)
        
        # เริ่มต้นระบบวัดประสิทธิภาพ
        logger.info("กำลังเริ่มต้นระบบวัดประสิทธิภาพ")
        self.metrics = RAGMetrics(log_dir=self.logs_dir)
        
        # พยายามโหลด vector store ถ้ามีอยู่แล้ว
        try:
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                self.vector_store.load_existing()
                # เรียกใช้เมธอดเริ่มต้น retriever โดยไม่ต้องผ่านเอกสาร
                self._init_retriever([])
        except Exception as e:
            logger.warning(f"ไม่สามารถโหลด vector store ที่มีอยู่: {e}")
        
        logger.info("เริ่มต้นส่วนประกอบเสร็จสมบูรณ์")
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        โหลดเอกสารจากไฟล์
        
        Args:
            file_path: ที่อยู่ไฟล์ที่ต้องการโหลด
            
        Returns:
            รายการเอกสารที่โหลดได้
        """
        logger.info(f"กำลังโหลดเอกสารจาก: {file_path}")
        
        # ตรวจสอบว่าไฟล์มีอยู่จริง
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ไม่พบไฟล์: {file_path}")
        
        # ดูนามสกุลไฟล์
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # โหลดเอกสารตามประเภทไฟล์
        if file_ext == ".txt":
            documents = DocumentLoader.load_text_file(file_path)
        elif file_ext == ".pdf":
            documents = DocumentLoader.load_pdf_file(file_path)
        elif file_ext == ".csv":
            documents = DocumentLoader.load_csv_file(file_path)
        elif file_ext == ".docx":
            documents = DocumentLoader.load_docx_file(file_path)
        else:
            raise ValueError(f"ไม่รองรับไฟล์ประเภท: {file_ext}")
        
        # เพิ่ม document_id ให้กับเอกสาร
        documents = add_document_ids(documents)
        
        # เก็บเอกสารไว้
        self.documents.extend(documents)
        
        logger.info(f"โหลดเอกสาร {len(documents)} รายการเสร็จสมบูรณ์")
        return documents
    
    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.txt", loader_cls=None) -> List[Document]:
        """
        โหลดเอกสารทั้งหมดในไดเร็กทอรี
        
        Args:
            directory_path: ที่อยู่ไดเร็กทอรี
            glob_pattern: รูปแบบชื่อไฟล์ที่ต้องการโหลด
            loader_cls: คลาสของ loader ที่ต้องการใช้ (ถ้าไม่ระบุจะใช้ค่าเริ่มต้นตามประเภทไฟล์)
            
        Returns:
            รายการเอกสารที่โหลดได้
        """
        logger.info(f"กำลังโหลดเอกสารจากไดเร็กทอรี: {directory_path} ด้วยรูปแบบ: {glob_pattern}")
        
        # ตรวจสอบว่าไดเร็กทอรีมีอยู่จริง
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"ไม่พบไดเร็กทอรี: {directory_path}")
        
        # โหลดเอกสารจากไดเร็กทอรี
        documents = DocumentLoader.load_directory(
            directory_path,
            glob_pattern=glob_pattern,
            loader_cls=loader_cls,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        # เพิ่ม document_id ให้กับเอกสาร
        documents = add_document_ids(documents)
        
        # เก็บเอกสารไว้
        self.documents.extend(documents)
        
        logger.info(f"โหลดเอกสาร {len(documents)} รายการเสร็จสมบูรณ์")
        return documents
    
    def load_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        โหลดเอกสารจากข้อความ
        
        Args:
            text: ข้อความที่ต้องการโหลด
            metadata: ข้อมูล metadata (ถ้ามี)
            
        Returns:
            รายการเอกสาร (1 รายการ)
        """
        logger.info("กำลังสร้างเอกสารจากข้อความ")
        
        # สร้างเอกสารจากข้อความ
        documents = DocumentLoader.load_from_text(text, metadata=metadata)
        
        # เพิ่ม document_id ให้กับเอกสาร
        documents = add_document_ids(documents)
        
        # เก็บเอกสารไว้
        self.documents.extend(documents)
        
        logger.info("สร้างเอกสารจากข้อความเสร็จสมบูรณ์")
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        ประมวลผลเอกสาร (แบ่งเป็นชั้นๆ)
        
        Args:
            documents: รายการเอกสารที่ต้องการประมวลผล
            
        Returns:
            รายการเอกสารที่ประมวลผลแล้ว
        """
        logger.info(f"กำลังประมวลผลเอกสาร {len(documents)} รายการ")
        
        # ตรวจสอบว่ามีเอกสารหรือไม่
        if not documents:
            logger.warning("ไม่มีเอกสารให้ประมวลผล")
            return []
        
        # แบ่งเอกสารเป็นชั้นๆ
        text_splitter = EnhancedTextSplitter()
        chunked_documents = text_splitter.split_documents(documents)
        
        # เพิ่ม document_id ให้กับชั้นที่แบ่ง
        chunked_documents = add_document_ids(chunked_documents)
        
        logger.info(f"แบ่งเอกสารเป็น {len(chunked_documents)} ชั้น")
        return chunked_documents
    
    def create_vector_store(self, documents: List[Document], recreate: bool = False) -> None:
        """
        สร้าง vector store จากเอกสาร
        
        Args:
            documents: รายการเอกสารที่ต้องการเก็บใน vector store
            recreate: สร้าง vector store ใหม่แทนที่จะเพิ่มเอกสาร
        """
        logger.info(f"กำลัง{'สร้าง' if recreate else 'เพิ่มเอกสารเข้า'} vector store")
        
        # ตรวจสอบว่ามีเอกสารหรือไม่
        if not documents:
            logger.warning("ไม่มีเอกสารให้เพิ่มใน vector store")
            return
        
        # สร้างหรือเพิ่มใน vector store
        if recreate or not os.path.exists(self.persist_dir):
            # สร้าง vector store ใหม่
            logger.info("กำลังสร้าง vector store ใหม่")
            self.vector_store.create_from_documents(documents)
        else:
            # โหลด vector store ที่มีอยู่แล้ว
            logger.info("กำลังโหลด vector store ที่มีอยู่แล้ว")
            self.vector_store.load_existing()
            
            # เพิ่มเอกสารใหม่
            logger.info(f"กำลังเพิ่มเอกสาร {len(documents)} รายการใน vector store")
            self.vector_store.add_documents(documents)
        
        logger.info("สร้าง/เพิ่มเอกสารใน vector store เสร็จสมบูรณ์")
        
        # เริ่มต้น retriever
        self._init_retriever(documents)
    
    def _init_retriever(self, documents: List[Document]) -> None:
        """
        เริ่มต้น retriever สำหรับการค้นคืน
        
        Args:
            documents: รายการเอกสารสำหรับ retriever
        """
        logger.info("กำลังเริ่มต้น retriever")
        
        # สร้าง enhanced retriever
        self.retriever = EnhancedRetriever(vector_store=self.vector_store)
        
        # สร้าง hybrid retriever ถ้าต้องการ
        if self.use_hybrid_search:
            logger.info("กำลังเริ่มต้น hybrid retriever")
            self.hybrid_retriever = HybridSearchRetriever(
                vector_store=self.vector_store,
                documents=documents
            )
        
        logger.info("เริ่มต้น retriever เสร็จสมบูรณ์")
    
    @timer
    def query(
        self,
        query: str,
        k: int = config.TOP_K,
        use_hybrid: Optional[bool] = None,
        use_self_critique: bool = True
    ) -> Dict[str, Any]:
        """
        ค้นหาและสร้างคำตอบจากคำถาม
        
        Args:
            query: คำถามที่ต้องการคำตอบ
            k: จำนวนเอกสารที่ต้องการค้นคืน
            use_hybrid: ใช้ hybrid search หรือไม่ (None = ใช้ค่าเริ่มต้นจาก self.use_hybrid_search)
            use_self_critique: ใช้การวิเคราะห์และปรับปรุงคำตอบด้วยตนเองหรือไม่
            
        Returns:
            ผลลัพธ์การค้นหาและคำตอบ
        """
        logger.info(f"กำลังค้นหาคำตอบสำหรับคำถาม: '{query}'")
        result = {"query": query}
        
        start_time = time.time()
        
        # เลือกว่าจะใช้ hybrid search หรือไม่
        use_hybrid_search = self.use_hybrid_search if use_hybrid is None else use_hybrid
        
        # ค้นคืนเอกสาร
        try:
            if use_hybrid_search and self.hybrid_retriever:
                logger.info("กำลังใช้ hybrid search")
                retrieved_docs = self.hybrid_retriever.hybrid_search(query, k=k)
            else:
                logger.info("กำลังใช้ enhanced retrieval")
                retrieved_docs = self.retriever.enhanced_retrieval(query, k=k)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการค้นคืนเอกสาร: {e}")
            retrieved_docs = []
        
        retrieval_end_time = time.time()
        retrieval_latency = retrieval_end_time - start_time
        
        # บันทึกประสิทธิภาพการค้นคืน
        self.metrics.log_retrieval_performance(
            query=query,
            retrieved_docs=retrieved_docs,
            latency_metrics={"latency_seconds": retrieval_latency}
        )
        
        result["retrieved_documents"] = retrieved_docs
        result["retrieval_time"] = retrieval_latency
        
        # สร้างบริบทจากเอกสารที่ค้นคืนได้
        context = self.prompt_manager.build_context_from_docs(retrieved_docs)
        
        # สร้างคำตอบ
        try:
            if use_self_critique:
                logger.info("กำลังสร้างคำตอบด้วยการวิเคราะห์และปรับปรุงด้วยตนเอง")
                response_data = self.prompt_manager.generate_with_self_critique(query, context)
                response = response_data["refined_response"]
                result["initial_response"] = response_data["initial_response"]
            else:
                logger.info("กำลังสร้างคำตอบแบบปกติ")
                response = self.prompt_manager.generate_rag_response(query, context)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}")
            response = f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}"
        
        generation_end_time = time.time()
        generation_latency = generation_end_time - retrieval_end_time
        total_latency = generation_end_time - start_time
        
        # บันทึกประสิทธิภาพการสร้างคำตอบ
        self.metrics.log_generation_performance(
            query=query,
            context=context,
            response=response,
            context_docs=retrieved_docs,
            latency_metrics={
                "latency_seconds": generation_latency,
                "total_latency_seconds": total_latency
            }
        )
        
        result["response"] = response
        result["generation_time"] = generation_latency
        result["total_time"] = total_latency
        
        logger.info(f"สร้างคำตอบเสร็จสมบูรณ์ ใช้เวลารวม {total_latency:.3f} วินาที")
        return result
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        รับข้อมูลของระบบ
        
        Returns:
            ข้อมูลของระบบ
        """
        # รับข้อมูล vector store
        vector_store_stats = {}
        if self.vector_store and hasattr(self.vector_store, "get_collection_stats"):
            vector_store_stats = self.vector_store.get_collection_stats()
        
        # สร้าง dictionary ข้อมูลระบบ
        system_info = {
            "data_dir": self.data_dir,
            "persist_dir": self.persist_dir,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "use_hybrid_search": self.use_hybrid_search,
            "loaded_documents": len(self.documents),
            "vector_store": vector_store_stats
        }
        
        return system_info
    
    def export_metrics(self) -> str:
        """
        ส่งออกข้อมูลประสิทธิภาพ
        
        Returns:
            ที่อยู่ไฟล์ที่ส่งออก
        """
        logger.info("กำลังส่งออกข้อมูลประสิทธิภาพ")
        
        # สร้างกราฟ
        self.metrics.plot_retrieval_metrics()
        self.metrics.plot_generation_metrics()
        
        # ส่งออก log
        output_path = self.metrics.export_logs()
        
        logger.info(f"ส่งออกข้อมูลประสิทธิภาพเสร็จสมบูรณ์ ไฟล์อยู่ที่: {output_path}")
        return output_path