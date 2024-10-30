import datetime
import os
import uuid
from typing import List


from langchain.retrievers import MultiVectorRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from dotenv import load_dotenv
from qwen_vl_utils import process_vision_info
from unstructured.partition.pdf import partition_pdf

# Load environment variables
load_dotenv()


class MultimodalRAG:
    def __init__(self, input_path: str,persist_directory: str, model_base_url: str = "http://192.168.123.178:8000/v1"):
        self.input_path = input_path
        self.output_path = os.path.join(input_path, "figures")
        self.model_base_url = model_base_url
        #self.model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct/"
        self.model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4/"
        os.makedirs(self.output_path, exist_ok=True)
        self.image_mapping = {}
        self.persist_directory = persist_directory

        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        #不适配baichuan
        #self.embeddings = OpenAIEmbeddings(model="Baichuan-Text-Embedding",api_key="sk-607cc5079f932337a013fc0fb0e06546",base_url="https://api.baichuan-ai.com/v1")

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )

        # Initialize Qwen2-VL model and processor
        self.model = self.init_model()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Initialize vector storage and retriever
        self.vectorstore = Chroma(
            collection_name="content",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.store = InMemoryStore()
        self.id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
            search_kwargs={"k": 2}
        )

    def init_model(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="half", device_map="auto")
        return model

    def invoke(self, messages):
        """Process messages and generate response using Qwen2-VL"""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else ""

    def parse_pdf(self, pdf_filename: str) -> tuple[List[str], List[str], List[str]]:
        """Parse PDF and extract text, tables, and images"""
        image_elements = []
        if os.path.exists(self.output_path):
            for image_file in os.listdir(self.output_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Found cached image: {image_file}")
                    image_path = os.path.join(self.output_path, image_file)
                    image_elements.append(image_path)

        extract_images = len(image_elements) == 0
        print(f"{'Need' if extract_images else 'No need'} to extract images")

        print(f"Starting PDF parsing...{datetime.datetime.now()}\n")

        raw_pdf_elements = partition_pdf(
            filename=os.path.join(self.input_path, pdf_filename),
            extract_images_in_pdf=extract_images,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=2000,
            new_after_n_chars=1800,
            combine_text_under_n_chars=1000,
            image_output_dir_path=self.output_path if extract_images else None,
        )
        print(f"Completed PDF parsing...{datetime.datetime.now()}\n")

        text_elements = []
        table_elements = []

        for element in raw_pdf_elements:
            if 'CompositeElement' in str(type(element)):
                text_elements.append(element.text)
                print(f"text_elements.text:{element.text}")
            elif 'Table' in str(type(element)):
                table_elements.append(element.text)
                print(f"table_elements.text:{element.text}")

        if extract_images and len(image_elements) == 0:
            for image_file in os.listdir(self.output_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Processing newly extracted image: {image_file}")
                    image_path = os.path.join(self.output_path, image_file)
                    image_elements.append(image_path)
        #text_elements_fake = []
        #table_elements_fake = []
        return text_elements, table_elements, image_elements

    def summarize_image(self, encoded_image: str) -> str:
        """Summarize image content"""
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": encoded_image
                    },
                    {
                        "type": "text",
                        "text": "描述这张图像的主要内容,120字以内。"
                    },
                ]
            }
        ]

        response = self.invoke(prompt)
        self.image_mapping[response] = encoded_image
        print(f"Image summary: {response}")
        return response

    def find_image_path(self, description: str) -> str:
        """Find the most similar image description and return its path"""
        if description in self.image_mapping:
            return self.image_mapping[description]
        for desc, path in self.image_mapping.items():
            if any(word in desc for word in description.split()):
                return path
        return None

    def process_context_with_images(self, contexts) -> list:
        """Process contexts and identify any image descriptions"""
        processed_contexts = []
        for doc in contexts:
            #content = doc.page_content
            image_path = self.find_image_path(doc)
            if image_path:
                processed_contexts.append({
                    "type": "image",
                    "path": image_path,
                    "description": doc
                })
            else:
                processed_contexts.append({
                    "type": "text",
                    "content": doc
                })
        return processed_contexts

    def ask_question(self, question: str):
        """Enhanced question answering with separate image and text processing"""
        try:
            print(f"question: {question}")
            contexts = self.retriever.invoke(question)
            print(f"self.retriever.invoke response: {contexts}\n")
            """
                self.retriever.invoke response: ['这张图像展示了不同模型在三个数据集（Boston、Energy、Yacht）上的性能比较，
                通过RMSE和时间（秒）的散点图来表示。每个数据集包含两个子图，分别表示RMSE和时间的关系。图中使用了不同的符号和颜色来区分不同
                的模型，如PCA+ESS、PCA+VI、SWAG等。通过观察散点图，可以直观地看出不同模型在不同数据集上的表现，以及它们在计算时间和预测精
                度之间的权衡。', '这张图像展示了不同模型在各种数据集上的训练过程中的对数似然函数随训练轮数的变化情况。图中包含了9个子图，
                每个子图对应一个数据集，包括Boston、Concrete、Energy、Kin8nm、Naval、Power、Wine、Yacht和Protein。每个子图中，
                不同颜色的线代表不同的模型，如TAGI-V、TAGI-V 2L、TAGI、PBP、MC-dropout、DVI、Ensemble和NN。通过观察这些曲线，可以了
                解不同模型在不同数据集上的性能变化趋势。']
                
            """
            # 分离图片描述和其他内容
            image_contexts = []
            text_contexts = []
            image_paths = []
            image_summaries = []

            for context in contexts:
                image_path = self.find_image_path(context)
                if image_path:
                    image_contexts.append(context)
                    image_paths.append(image_path)
                    image_summaries.append(context)  # 使用图片描述作为摘要
                else:
                    text_contexts.append(context)

            #processed_contexts = self.process_context_with_images(contexts)

            message_content = []
            message_content.append({
                "type": "text",
                "text": f"请根据以下内容回答问题。问题是：{question},如果问题是英文，则英文回答，如果是中文，中文回答\n\n"
            })

            # 添加图片内容
            for i, (path, summary) in enumerate(zip(image_paths, image_summaries)):
                message_content.extend([
                    {
                        "type": "image",
                        "image": path
                    },
                    {
                        "type": "text",
                        "text": f"图片 {i + 1} 描述：{summary}\n"
                    }
                ])

            """
            for item in processed_contexts:
                if item["type"] == "image":
                    message_content.append({
                        "type": "image",
                        "image": item["path"]
                    })
                    message_content.append({
                        "type": "text",
                        "text": f"图片描述：{item['description']}"
                    })
                else:
                    message_content.append({
                        "type": "text",
                        "text": item["content"]
                    })
            """
            # 添加文本内容
            if text_contexts:
                message_content.append({
                    "type": "text",
                    "text": "\n相关文本内容：\n" + "\n".join(text_contexts)
                })

            messages = [{
                "role": "user",
                "content": message_content
            }]

            response = self.invoke(messages)

            return {
                "answer": response,
                "images": image_paths,
                "image_summaries": image_summaries,
                "text_contexts": text_contexts
            }


        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return {
                "answer": f"Sorry, an error occurred while processing your question: {str(e)}",
                "images": [],
                "image_summaries": [],
                "text_contexts": []
            }

    def add_documents_to_retriever(self, contents: List[str], is_image_summary: bool = False):
        """Add documents to the retriever"""
        doc_ids = [str(uuid.uuid4()) for _ in contents]
        docs = [
            Document(page_content=content, metadata={
                self.id_key: doc_ids[i],
                "type": "image_summary" if is_image_summary else "content"
            })
            for i, content in enumerate(contents)
        ]
        self.retriever.vectorstore.add_documents(docs)# 存入chrome向量数据库
        self.retriever.docstore.mset(list(zip(doc_ids, contents)))
        """
            [(id1, content1), (id2, content2), ..., (idn, contentn)]，其中id1, id2, …, idn是文档的ID，而content1, content2, …, contentn是与这些ID相关联的内容。
        """

    def process_document(self, pdf_filename: str):
        """Process document and prepare for retrieval"""
        print(f"Starting document processing: {pdf_filename}")

        try:
            print("Parsing PDF...")
            text_elements, table_elements, image_elements = self.parse_pdf(pdf_filename)

            # Add text and table content directly to retriever
            print(f"Adding {len(text_elements)} text elements to retriever...")
            if text_elements:
                self.add_documents_to_retriever(text_elements)

            print(f"Adding {len(table_elements)} table elements to retriever...")
            if table_elements:
                self.add_documents_to_retriever(table_elements)

            # Process images and add their summaries
            print(f"Processing {len(image_elements)} image elements...")
            image_summaries = []
            for image in image_elements:
                try:
                    summary = self.summarize_image(image)
                    image_summaries.append(summary)
                    print(f"✓ Successfully processed image: {summary}")
                except Exception as e:
                    print(f"Error processing image: {str(e)}")

            if image_summaries:
                self.add_documents_to_retriever(image_summaries, is_image_summary=True)

            print("Document processing complete!")

            #self.vectorstore.persist()

            return {
                "text_elements": len(text_elements),
                "table_elements": len(table_elements),
                "image_elements": len(image_elements),
                "image_summaries": len(image_summaries)
            }

        except Exception as e:
            print(f"Error during document processing: {str(e)}")
            raise


def main():
    """Main function"""
    try:
        print("Initializing RAG system...")
        rag = MultimodalRAG(input_path=os.getcwd())

        # Test text inference
        print("Testing text inference...")
        messages = [{"role": "user", "content": "请告诉我你是谁？"}]
        response = rag.invoke(messages)
        print(f"Test result: {response}")

        # Test image recognition
        print(f"Testing image recognition...")
        test_image_path = f"{rag.output_path}/figure-28-13.jpg"
        if os.path.exists(test_image_path):
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{test_image_path}"
                        },
                        {
                            "type": "text",
                            "text": "描述这张图像的主要内容，120个字以内。"
                        },
                    ]
                }
            ]
            response = rag.invoke(prompt)
            print(f"Image test result: {response}")
        else:
            print("Test image not found")

        # Process document
        print("\nProcessing document...")
        stats = rag.process_document("TAGIV.pdf")
        print("\nProcessing statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # Test query
        print("\nExecuting test query...")
        question = "我需要对比不同模型在不同数据集上的RMSE（均方根误差）随时间变化的趋势，请给出建议和示意图"
        result = rag.ask_question(question)
        print(f"Answer: {result['answer']}")
        if result['images']:
            print(f"Retrieved images: {result['images']}")

    except Exception as e:
        print(f"\nProgram execution error: {str(e)}")
        raise


if __name__ == "__main__":
    main()