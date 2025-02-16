# TDS-Project-1: Automation Agent

An AI-powered **automation agent** designed for operational and business automation tasks. This project utilizes **LLMs, Embeddings, SQLite, and more** to execute automation tasks efficiently.

## 📌 Features

- **Operations Automation:** Perform text processing, file reading, markdown conversion, and other tasks.
- **Business Automation:** Fetch API data, perform SQL queries, handles Git operations, web scraping, and image processing.
- **Secure Execution:** Ensures **data security** by restricting unauthorized data deletion or exfiltration.
- **FastAPI API:** Exposes endpoints to execute automation tasks using **plain-English commands**.

## 🛠 Installation & Setup

**Clone the Repository**

```sh
git clone https://github.com/sadiya125/automation-agent.git
cd automation-agent
```

**Run with Podman**

```sh
podman build -t sadiya125/automation-agent .
podman run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 sadiya125/automation-agent
```

(OR)

**Run with Docker**

```sh
docker build -t sadiya125/automation-agent .
docker run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 sadiya125/automation-agent
```

## 🚀 Usage

### API Endpoints

Once running, access the API at:

```bash
http://localhost:8000/docs
```

You can test endpoints like:

```sh
curl http://localhost:8000/read?path=/data/format.md
```

## 📦 Project Structure

```bash
/app
├── app.py        # FastAPI Application
├── tasksA.py     # Operational Automation Tasks
├── tasksB.py     # Business Automation Tasks
├── requirements.txt  # Python Dependencies
├── Dockerfile    # Containerization Setup
└── README.md     # Project Documentation
```

## 🤖 AI Proxy Configuration

This project uses AI Proxy for LLM-based tasks.
Ensure you have a valid AIPROXY_TOKEN from:

```arduino
https://aiproxy.sanand.workers.dev/
```

## 📜 License

This project is Open-Source under the MIT License.

#### 🚀 Happy Automating! 🤖✨
