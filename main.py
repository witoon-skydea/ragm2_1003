import argparse
import os
import sys
import time
import traceback
from pathlib import Path
import logging
from src.rag_system import ThaiRAGSystem
import config

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("main.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("main")

def process_documents(rag_system, args):
    """
    ประมวลผลเอกสารและเพิ่มลงใน vector store
    
    Args:
        rag_system: ระบบ RAG
        args: พารามิเตอร์การรัน
    """
    # โหลดเอกสาร
    documents = []
    
    if args.file:
        for file_path in args.file:
            try:
                docs = rag_system.load_documents(file_path)
                documents.extend(docs)
                logger.info(f"โหลดเอกสารจาก {file_path} สำเร็จ: {len(docs)} รายการ")
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ {file_path}: {e}")
    
    if args.dir:
        for dir_path in args.dir:
            try:
                # เตรียม loader classes
                from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
                
                # ถ้ามีการระบุนามสกุลไฟล์ที่ต้องการโหลด
                if args.ext:
                    ext = args.ext.lower()
                    glob_pattern = f"**/*.{ext}"
                    
                    loader_cls = TextLoader  # ค่าเริ่มต้น
                    if ext == "pdf":
                        loader_cls = PyPDFLoader
                    elif ext == "csv":
                        loader_cls = CSVLoader
                    elif ext == "docx":
                        loader_cls = Docx2txtLoader
                    
                    # ตรวจสอบไฟล์ในไดเร็กทอรีก่อน
                    matched_files = list(Path(dir_path).glob(glob_pattern))
                    logger.info(f"พบไฟล์ .{ext} ในไดเร็กทอรี: {len(matched_files)} ไฟล์")
                    
                    if not matched_files:
                        logger.warning(f"ไม่พบไฟล์ .{ext} ในไดเร็กทอรี {dir_path}")
                        continue
                    
                    # โหลดเอกสาร
                    docs = rag_system.load_directory(
                        dir_path, 
                        glob_pattern=glob_pattern,
                        loader_cls=loader_cls
                    )
                else:
                    # ถ้าไม่ได้ระบุนามสกุลไฟล์ ให้โหลดทุกประเภทไฟล์ที่รองรับ
                    all_docs = []
                    # รายการนามสกุลไฟล์ที่รองรับ
                    supported_exts = {
                        "txt": TextLoader,
                        "pdf": PyPDFLoader,
                        "csv": CSVLoader,
                        "docx": Docx2txtLoader
                    }
                    
                    for ext, loader_cls in supported_exts.items():
                        try:
                            # ตรวจสอบไฟล์ในไดเร็กทอรีก่อน
                            matched_files = list(Path(dir_path).glob(f"**/*.{ext}"))
                            logger.info(f"พบไฟล์ .{ext} ในไดเร็กทอรี: {len(matched_files)} ไฟล์")
                            
                            # ถ้าไม่พบไฟล์ให้ข้ามไป
                            if not matched_files:
                                logger.info(f"ไม่พบไฟล์ .{ext} ในไดเร็กทอรี {dir_path} จึงข้ามไป")
                                continue
                                
                            glob_pattern = f"**/*.{ext}"
                            logger.info(f"กำลังพยายามโหลดไฟล์ .{ext} จาก {dir_path} ด้วย {loader_cls.__name__}")
                            ext_docs = rag_system.load_directory(
                                dir_path,
                                glob_pattern=glob_pattern,
                                loader_cls=loader_cls
                            )
                            logger.info(f"โหลดไฟล์ .{ext} จาก {dir_path} สำเร็จ: {len(ext_docs)} รายการ")
                            all_docs.extend(ext_docs)
                        except Exception as ext_error:
                            logger.warning(f"ไม่สามารถโหลดไฟล์ .{ext} จาก {dir_path}: {ext_error}")
                            logger.debug(f"รายละเอียดข้อผิดพลาด: {traceback.format_exc()}")
                    
                    if len(all_docs) == 0:
                        logger.warning(f"ไม่พบไฟล์ที่รองรับ (txt, pdf, csv, docx) ในไดเร็กทอรี {dir_path}")
                        continue
                    
                    logger.info(f"โหลดไฟล์ทั้งหมดจาก {dir_path} สำเร็จรวม: {len(all_docs)} รายการ")
                    docs = all_docs
                    
                documents.extend(docs)
                logger.info(f"โหลดเอกสารจากไดเร็กทอรี {dir_path} สำเร็จ: {len(docs)} รายการ")
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการโหลดไดเร็กทอรี {dir_path}: {e}")
                logger.error(traceback.format_exc())
    
    if args.text:
        try:
            docs = rag_system.load_text(args.text)
            documents.extend(docs)
            logger.info(f"สร้างเอกสารจากข้อความสำเร็จ: {len(docs)} รายการ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างเอกสารจากข้อความ: {e}")
    
    if not documents:
        logger.warning("ไม่มีเอกสารให้ประมวลผล")
        return
    
    # ประมวลผลเอกสาร
    try:
        chunked_docs = rag_system.process_documents(documents)
        logger.info(f"ประมวลผลเอกสารสำเร็จ: {len(chunked_docs)} ชั้น")
        
        # สร้างหรือเพิ่มใน vector store
        rag_system.create_vector_store(chunked_docs, recreate=args.recreate)
        logger.info("เพิ่มเอกสารลงใน vector store สำเร็จ")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการประมวลผลเอกสาร: {e}")
        logger.error(traceback.format_exc())


def query_mode(rag_system, args):
    """
    โหมดตอบคำถาม
    
    Args:
        rag_system: ระบบ RAG
        args: พารามิเตอร์การรัน
    """
    if args.query:
        # ตอบคำถามเดียว
        try:
            result = rag_system.query(
                args.query, 
                k=args.top_k, 
                use_hybrid=args.hybrid,
                use_self_critique=not args.no_critique
            )
            
            # แสดงผลลัพธ์
            print("\n" + "="*80)
            print(f"คำถาม: {result['query']}")
            print("="*80)
            
            if 'initial_response' in result:
                print("\nคำตอบเบื้องต้น:")
                print("-"*80)
                print(result['initial_response'])
                print("-"*80)
                print("\nคำตอบที่ปรับปรุงแล้ว:")
                
            print("-"*80)
            print(result['response'])
            print("-"*80)
            
            print(f"\nเวลาที่ใช้: รวม {result['total_time']:.3f} วินาที")
            print(f"- การค้นคืน: {result['retrieval_time']:.3f} วินาที")
            print(f"- การสร้างคำตอบ: {result['generation_time']:.3f} วินาที")
            
            # แสดงเอกสารที่ค้นคืนได้ถ้าต้องการ
            if args.show_docs:
                print("\nเอกสารที่ค้นคืนได้:")
                for i, doc in enumerate(result['retrieved_documents']):
                    print(f"\n--- เอกสาร {i+1} ---")
                    
                    # แสดงแหล่งที่มาถ้ามี
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        print(f"แหล่งที่มา: {doc.metadata['source']}")
                    
                    # แสดงเนื้อหา (จำกัดความยาว)
                    max_length = 300
                    content = doc.page_content
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    print(f"เนื้อหา: {content}")
            
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตอบคำถาม: {e}")
            logger.error(traceback.format_exc())
    
    elif args.interactive:
        # โหมดโต้ตอบ
        print("\n" + "="*80)
        print("โหมดโต้ตอบกับระบบ RAG ภาษาไทย")
        print("พิมพ์ 'q' หรือ 'exit' เพื่อออกจากโปรแกรม")
        print("="*80 + "\n")
        
        while True:
            try:
                query = input("\nคำถาม: ")
                
                if query.lower() in ['q', 'exit', 'quit']:
                    print("จบการทำงาน")
                    break
                
                if not query.strip():
                    continue
                
                start_time = time.time()
                result = rag_system.query(
                    query, 
                    k=args.top_k, 
                    use_hybrid=args.hybrid,
                    use_self_critique=not args.no_critique
                )
                end_time = time.time()
                
                print("\nคำตอบ:")
                print("-"*80)
                print(result['response'])
                print("-"*80)
                
                print(f"เวลาที่ใช้: {end_time - start_time:.3f} วินาที")
                
                # แสดงเอกสารที่ค้นคืนได้ถ้าต้องการ
                if args.show_docs:
                    print("\nเอกสารที่ค้นคืนได้:")
                    for i, doc in enumerate(result['retrieved_documents']):
                        print(f"\n--- เอกสาร {i+1} ---")
                        
                        # แสดงแหล่งที่มาถ้ามี
                        if hasattr(doc, "metadata") and "source" in doc.metadata:
                            print(f"แหล่งที่มา: {doc.metadata['source']}")
                        
                        # แสดงเนื้อหา (จำกัดความยาว)
                        max_length = 200
                        content = doc.page_content
                        if len(content) > max_length:
                            content = content[:max_length] + "..."
                        print(f"เนื้อหา: {content}")
                
            except KeyboardInterrupt:
                print("\nยกเลิกโดยผู้ใช้")
                break
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาด: {e}")
                logger.error(traceback.format_exc())
                print(f"เกิดข้อผิดพลาด: {e}")


def main():
    """ฟังก์ชัน main"""
    parser = argparse.ArgumentParser(description="ระบบ RAG ชั้นสูงสำหรับภาษาไทย")
    
    # พารามิเตอร์ทั่วไป
    parser.add_argument("--data-dir", type=str, help="ไดเร็กทอรีข้อมูล")
    parser.add_argument("--persist-dir", type=str, help="ไดเร็กทอรีสำหรับบันทึก vector store")
    parser.add_argument("--embedding-model", type=str, help="ชื่อโมเดล embeddings")
    parser.add_argument("--llm-model", type=str, help="ชื่อโมเดล LLM")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="จำนวนเอกสารที่จะค้นคืน")
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false", help="ไม่ใช้ hybrid search")
    parser.add_argument("--no-critique", action="store_true", help="ไม่ใช้การวิเคราะห์และปรับปรุงคำตอบด้วยตนเอง")
    parser.add_argument("--show-docs", action="store_true", help="แสดงเอกสารที่ค้นคืนได้")
    
    # โหมดการทำงาน
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--process", action="store_true", help="โหมดประมวลผลเอกสาร")
    mode_group.add_argument("--query", type=str, help="คำถามที่ต้องการคำตอบ")
    mode_group.add_argument("--interactive", action="store_true", help="โหมดโต้ตอบ")
    mode_group.add_argument("--info", action="store_true", help="แสดงข้อมูลระบบ")
    
    # พารามิเตอร์สำหรับการประมวลผลเอกสาร
    parser.add_argument("--file", nargs="+", help="ไฟล์ที่ต้องการโหลด")
    parser.add_argument("--dir", nargs="+", help="ไดเร็กทอรีที่ต้องการโหลด")
    parser.add_argument("--ext", type=str, help="นามสกุลไฟล์ที่ต้องการโหลดจากไดเร็กทอรี (ถ้าไม่ระบุจะโหลดทุกไฟล์ที่รองรับ ได้แก่ txt, pdf, csv, docx)")
    parser.add_argument("--text", type=str, help="ข้อความที่ต้องการโหลดโดยตรง")
    parser.add_argument("--recreate", action="store_true", help="สร้าง vector store ใหม่แทนที่จะเพิ่มเอกสาร")
    
    args = parser.parse_args()
    
    # ตรวจสอบว่ามีโหมดการทำงานหรือไม่
    if not (args.process or args.query or args.interactive or args.info):
        # ถ้าไม่มีโหมดแต่มีการระบุ --file, --dir, หรือ --text ให้สันนิษฐานว่าเป็นโหมดประมวลผล
        if args.file or args.dir or args.text:
            args.process = True
        else:
            # ถ้าไม่มีโหมดและไม่มีพารามิเตอร์อื่น ให้ใช้โหมดโต้ตอบเป็นค่าเริ่มต้น
            args.interactive = True
    
    try:
        # สร้างระบบ RAG
        rag_system = ThaiRAGSystem(
            data_dir=args.data_dir,
            persist_dir=args.persist_dir,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            use_hybrid_search=args.hybrid
        )
        
        # ทำงานตามโหมดที่เลือก
        if args.process or args.file or args.dir or args.text:
            # โหมดประมวลผลเอกสาร
            process_documents(rag_system, args)
        
        elif args.query or args.interactive:
            # โหมดตอบคำถาม
            query_mode(rag_system, args)
        
        elif args.info:
            # แสดงข้อมูลระบบ
            info = rag_system.get_system_info()
            print("\n" + "="*80)
            print("ข้อมูลระบบ RAG ภาษาไทย")
            print("="*80)
            
            for key, value in info.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
            
            print("="*80 + "\n")
            
        # ส่งออกข้อมูลประสิทธิภาพ
        if hasattr(rag_system, "metrics") and rag_system.metrics.performance_logs:
            rag_system.export_metrics()
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการทำงาน: {e}")
        logger.error(traceback.format_exc())
        print(f"เกิดข้อผิดพลาด: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()