from flask import Flask, request, jsonify, session, flash, render_template
from flask_session import Session
import google.generativeai as genai
import logging
import re
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.corpus import stopwords
import nltk
import os
import joblib,json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from pathlib import Path

# Ensure NLTK data is downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.error(f"Failed to download NLTK data: {e}")
            raise RuntimeError("NLTK data download failed. Check network connectivity.")

download_nltk_data()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_PERMANENT'] = False
Session(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API
try:
    genai.configure(api_key='AIzaSyAN5rU9-qHNGFz2ChZh_LIwwybqNEXr7tI')
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logging.error(f"Failed to configure Gemini API: {e}")
    raise RuntimeError("Gemini API configuration failed.")

# Knowledge base
knowledge_base = [
    "DDoS attacks overwhelm network resources with excessive traffic, causing service disruptions in IoT and traditional networks.",
    "Packet sniffing captures unencrypted data on networks, compromising sensitive information in IoT devices and network infrastructure.",
    "IoT botnets, like Mirai, exploit vulnerable devices to launch coordinated attacks, impacting network availability.",
    "Man-in-the-middle attacks intercept network communications to steal data or inject malicious code, especially in unsecured IoT ecosystems.",
    "Weak IoT device passwords enable brute-force attacks, leading to unauthorized access and potential network compromise.",
    "IoT networks rely on interconnected devices communicating via protocols like MQTT, CoAP, or HTTP, often with limited security.",
    "Studies show 70% of IoT devices lack robust encryption, increasing vulnerability to attacks (e.g., HP IoT Security Study, 2014).",
    "Network segmentation can mitigate attack spread in IoT networks by isolating compromised devices.",
    "Research indicates DDoS attacks on IoT networks increased by 91% from 2016-2020 (Nokia Threat Intelligence Report, 2020).",
    "Zero-day exploits in IoT firmware pose significant risks, as patches are often delayed or unavailable."
]

def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    try:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'#+\s*', '', text)  # Remove headers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links
        text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)  # Remove code blocks
        text = re.sub(r'`(.*?)`', r'\1', text)  # Remove inline code
        text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)  # Remove blockquotes
        text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)  # Remove lists
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Remove numbered lists
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)  # Remove horizontal rules
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean multiple newlines
        return text.strip()
    except Exception as e:
        logging.error(f"Error in clean_markdown: {e}")
        return text

def convert_paragraph_to_points(paragraph: str, num_points: int = 5) -> list:
    if not paragraph or len(paragraph.strip()) < 10:
        return ["No valid response content to process."] + [''] * (num_points - 1)
    try:
        sentences = sent_tokenize(paragraph)
        if not sentences:
            return ["No sentences found in response."] + [''] * (num_points - 1)
        words = word_tokenize(paragraph.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        if not filtered_words:
            return ["No meaningful words found in response."] + [''] * (num_points - 1)
        freq_dist = FreqDist(filtered_words)
        sentence_scores = {}
        for sentence in sentences:
            sentence_word_tokens = word_tokenize(sentence.lower())
            sentence_word_tokens = [word for word in sentence_word_tokens if word.isalnum()]
            score = sum(freq_dist.get(word, 0) for word in sentence_word_tokens)
            sentence_scores[sentence] = score
        sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        key_points = sorted_sentences[:num_points]
        return key_points + [''] * (num_points - len(key_points))
    except Exception as e:
        logging.error(f"Error in convert_paragraph_to_points: {e}")
        return [f"Error processing response: {str(e)}"] + [''] * (num_points - 1)

@app.route('/')
def index():
    session['chat_history'] = session.get('chat_history', [])
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index.html: {e}")
        return jsonify({'error': 'Failed to load the page.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Validate message relevance
        relevant_keywords = ['network', 'iot', 'attack', 'ddos', 'sniffing', 'botnet', 'security', 'study', 'research',
                            'protocol', 'encryption']
        if not any(keyword in user_message.lower() for keyword in relevant_keywords):
            response = "Please ask about IoT networks, network attacks, or related studies (e.g., DDoS, botnets, IoT security research)."
            session['chat_history'] = session.get('chat_history', []) + [
                {'role': 'user', 'content': user_message},
                {'role': 'bot', 'content': response}
            ]
            flash('Response generated successfully!', 'success')
            return jsonify({'response': response})

        # RAG: Augment prompt with knowledge base
        rag_context = "\n".join(knowledge_base)
        prompt = f"""As an expert in IoT networks, network attacks, and related studies, provide a concise response to the user input below, focusing exclusively on IoT networks, network attacks, or relevant studies. Use the provided context to enhance accuracy:

**Context**: {rag_context}

**User Input**: "{user_message}"

**Deliverables**:
- **Response Focus**: Address the user's query with details on IoT networks, network attacks, or studies.
- **Key Information**: Highlight specific protocols, attack types, or study findings relevant to the input.
- **Contextual Use**: Suggest a practical scenario or application (e.g., securing IoT devices, analyzing attack impacts).
- **Technical Details**: Include one technical aspect (e.g., protocol, encryption method, attack vector).
- **Source Reference**: If applicable, reference a study or concept from the context (e.g., Mirai botnet, Nokia Report).

**Format**:
- Use concise bullet points for clarity.
- Limit response to 500 tokens for brevity.
- Ensure all content is relevant to IoT networks, network attacks, or studies (e.g., DDoS, botnets, IoT security research).
"""

        response = model.generate_content(
            prompt.format(rag_context=rag_context, user_message=user_message[:1000]),
            generation_config={"max_output_tokens": 10000}
        )
        if not response.text:
            raise ValueError("Empty response from Gemini API")

        cleaned_response = clean_markdown(response.text)
        response_points = convert_paragraph_to_points(cleaned_response, num_points=5)

        session['response_points'] = response_points
        session['chat_history'] = session.get('chat_history', []) + [
            {'role': '', 'content': user_message},
            {'role': '', 'content': '\n'.join(response_points)}
        ]
        flash('Response generated successfully!', 'success')
        return jsonify({'response': '\n'.join(response_points)})

    except Exception as e:
        logging.error(f"Gemini response generation error: {e}")
        error_message = f"Error: Could not generate response: {str(e)}"
        session['response_points'] = [error_message]
        session['chat_history'] = session.get('chat_history', []) + [
            {'role': 'user', 'content': user_message},
            {'role': 'bot', 'content': error_message}
        ]
        flash('Error generating response.', 'error')
        return jsonify({'error': error_message}), 500

@app.route('/attack', methods=['GET', 'POST'])
def attacks():
    return render_template('attack.html')

@app.route('/attack_submit', methods=['GET', 'POST'])
def attacks_submits():
    if request.method == 'POST':
        try:
            # Collect form inputs
            bwd_bulk_bytes = request.form['bwd_bulk_bytes']
            fwd_bulk_packets = request.form['fwd_bulk_packets']
            bwd_bulk_packets = request.form['bwd_bulk_packets']
            fwd_bulk_rate = request.form['fwd_bulk_rate']
            bwd_bulk_rate = request.form['bwd_bulk_rate']
            active_min = request.form['active_min']
            active_max = request.form['active_max']
            active_tot = request.form['active_tot']
            active_avg = request.form['active_avg']
            active_std = request.form['active_std']
            idle_min = request.form['idle_min']
            idle_max = request.form['idle_max']
            idle_tot = request.form['idle_tot']
            idle_avg = request.form['idle_avg']
            idle_std = request.form['idle_std']
            fwd_init_window_size = request.form['fwd_init_window_size']
            bwd_init_window_size = request.form['bwd_init_window_size']
            fwd_last_window_size = request.form['fwd_last_window_size']

            # Create dictionary of inputs
            arr = {
                'backward_bulk_bytes': float(bwd_bulk_bytes),
                'forward_bulk_packets': float(fwd_bulk_packets),
                'backward_bulk_packets': float(bwd_bulk_packets),
                'forward_bulk_rate': float(fwd_bulk_rate),
                'backward_bulk_rate': float(bwd_bulk_rate),
                'active_min': float(active_min),
                'active_max': float(active_max),
                'active_tot': float(active_tot),
                'active_avg': float(active_avg),
                'active_std': float(active_std),
                'idle_min': float(idle_min),
                'idle_max': float(idle_max),
                'idle_tot': float(idle_tot),
                'idle_avg': float(idle_avg),
                'idle_std': float(idle_std),
                'forward_init_window_size': float(fwd_init_window_size),
                'backward_init_window_size': float(bwd_init_window_size),
                'fwd_last_window_size': float(fwd_last_window_size)
            }

            attacks = ['ARP_Poisoning', 'DDOS_SlowLoris', 'DOS_SYN_Hping', 'MQTT_Publish', 'Metasploit_Brute_Force_SSH',
                       'Nmap_FIN_SCAN', 'NMAP_OS_DETECTION', 'NMAP_TCP_SCAN', 'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN']

            # Assuming knowledge_base is defined elsewhere
            rag_context = "\n".join(knowledge_base)

            # Updated prompt to predict attack type and explain reasoning
            prompt = """As an expert in IoT networks and network attacks, analyze the provided network traffic features to predict the most likely attack type from the following list: {attacks}. Provide a concise response with the following deliverables:

            **Context**: {rag_context}
            **User Input**: {input_features}

            **Deliverables**:
            - **Predicted Attack**: Identify the most likely attack type based on the input features.
            - **Reasoning**: Explain why this attack is likely, referencing specific feature values (e.g., high backward_bulk_bytes, low idle_min).
            - **Contextual Use**: Suggest a practical scenario or mitigation strategy for this attack in an IoT network.
            - **Technical Details**: Highlight one technical aspect (e.g., protocol, attack vector) relevant to the predicted attack.
            - **Source Reference**: If applicable, reference a study or concept from the context (e.g., Mirai botnet, Nokia Report).

            **Format**:
            - Use concise bullet points for clarity.
            - Limit response to 500 tokens for brevity.
            - Ensure all content is relevant to IoT networks and the specified attacks.
            """

            # Format prompt with inputs
            formatted_prompt = prompt.format(
                rag_context=rag_context,
                attacks=', '.join(attacks),
                input_features=json.dumps(arr, indent=2)
            )

            # Generate response using Gemini model (assuming model is defined)
            response = model.generate_content(
                formatted_prompt,
                generation_config={"max_output_tokens": 500}
            )
            if not response.text:
                raise ValueError("Empty response from Gemini API")

            # Clean and format response
            cleaned_response = clean_markdown(response.text)
            response_points = convert_paragraph_to_points(cleaned_response, num_points=5)

            # Store in session
            session['response_points'] = response_points
            session['chat_history'] = session.get('chat_history', []) + [
                {'role': '', 'content': '\n'.join(response_points)}
            ]
            flash('Response generated successfully!', 'success')
            return jsonify({'response': '\n'.join(response_points)})

        except KeyError as e:
            flash(f'Missing form field: {str(e)}', 'error')
            return jsonify({'error': f'Missing form field: {str(e)}'}), 400
        except ValueError as e:
            flash(f'Invalid input: {str(e)}', 'error')
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            flash(f'Error processing request: {str(e)}', 'error')
            return jsonify({'error': 'Internal server error'}), 500

@app.route('/protocol', methods=['GET', 'POST'])
def protocols():
    return render_template('protocol.html')

@app.route('/protocol_pred', methods=['GET', 'POST'])
def pred_prot():
    if request.method == 'POST':
        try:
            # Collect form inputs
            id_orig_p = request.form['id_orig_p']
            id_resp_p = request.form['id_resp_p']
            fwd_data_pkts_tot = request.form['fwd_data_pkts_tot']
            bwd_data_pkts_tot = request.form['bwd_data_pkts_tot']
            bwd_pkts_payload_min = request.form['bwd_pkts_payload_min']
            bwd_pkts_payload_max = request.form['bwd_pkts_payload_max']
            bwd_pkts_payload_tot = request.form['bwd_pkts_payload_tot']
            bwd_pkts_payload_avg = request.form['bwd_pkts_payload_avg']
            flow_pkts_payload_min = request.form['flow_pkts_payload_min']
            bwd_subflow_bytes = request.form['bwd_subflow_bytes']

            # Create dictionary of inputs
            arr = {
                'id_orig_p': float(id_orig_p),
                'id_resp_p': float(id_resp_p),
                'fwd_data_packets_tot': float(fwd_data_pkts_tot),
                'backward_data_packets_tot': float(bwd_data_pkts_tot),
                'bwd_packets_payload_minimum': float(bwd_pkts_payload_min),
                'backward_packets_payload_maximum': float(bwd_pkts_payload_max),
                'backward_packets_payload_tot': float(bwd_pkts_payload_tot),
                'backwards_packets_payload_average': float(bwd_pkts_payload_avg),
                'flow_packets_payload_minimum': float(flow_pkts_payload_min),
                'backward_subflow_bytes': float(bwd_subflow_bytes)
            }

            protocols = ['ICMP Protocol', 'TCP Protocol', 'UDP Protocol']

            # Assuming knowledge_base is defined elsewhere
            rag_context = "\n".join(knowledge_base)

            # Updated prompt to predict protocol and explain reasoning
            prompt = """As an expert in IoT networks and network protocols, analyze the provided network traffic features to predict the most likely protocol from the following list: {protocols}. Provide a concise response with the following deliverables:

            **Context**: {rag_context}
            **User Input**: {input_features}

            **Deliverables**:
            - **Predicted Protocol**: Identify the most likely protocol (ICMP, TCP, or UDP) based on the input features.
            - **Reasoning**: Explain why this protocol is likely, referencing specific feature values (e.g., high id_orig_p, low bwd_packets_payload_minimum).
            - **Contextual Use**: Suggest a practical scenario or application for this protocol in an IoT network (e.g., securing IoT devices, optimizing data transfer).
            - **Technical Details**: Highlight one technical aspect (e.g., protocol characteristics, packet structure, or vulnerabilities).
            - **Source Reference**: If applicable, reference a study or concept from the context (e.g., Mirai botnet, Nokia Report).

            **Format**:
            - Use concise bullet points for clarity.
            - Limit response to 500 tokens for brevity.
            - Ensure all content is relevant to IoT networks and the specified protocols.
            """

            # Format prompt with inputs
            formatted_prompt = prompt.format(
                rag_context=rag_context,
                protocols=', '.join(protocols),
                input_features=json.dumps(arr, indent=2)
            )

            # Generate response using Gemini model (assuming model is defined)
            response = model.generate_content(
                formatted_prompt,
                generation_config={"max_output_tokens": 500}
            )
            if not response.text:
                raise ValueError("Empty response from Gemini API")

            # Clean and format response
            cleaned_response = clean_markdown(response.text)
            response_points = convert_paragraph_to_points(cleaned_response, num_points=5)

            # Store in session
            session['response_points'] = response_points
            session['chat_history'] = session.get('chat_history', []) + [
                {'role': '', 'content': '\n'.join(response_points)}
            ]
            flash('Response generated successfully!', 'success')
            return jsonify({'response': '\n'.join(response_points)})

        except KeyError as e:
            flash(f'Missing form field: {str(e)}', 'error')
            return jsonify({'error': f'Missing form field: {str(e)}'}), 400
        except ValueError as e:
            flash(f'Invalid input: {str(e)}', 'error')
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            flash(f'Error processing request: {str(e)}', 'error')
            return jsonify({'error': 'Internal server error'}), 500

@app.route('/xai',methods=['GET','POST'])
def xai_pred():
    return render_template('xai.html')

@app.route('/xai/ext_proto',methods=['GET','POST'])
def xai():
    return render_template('ext_proto_explanation.html')

@app.route('/xai/rf_proto',methods=['GET','POST'])
def rf_prot():
    return render_template('rf_proto_explanation.html')

@app.route('/xai/lr_attack',methods=['GET','POST'])
def lr_attack():
    return render_template('lr_attack_explanation.html')

@app.route('/xai/rf_attack',methods=['GET','POST'])
def rf_attack():
    return render_template('rf_attack_explanation.html')

if __name__ == '__main__':
    # Ensure session directory exists
    session_dir = os.path.join(os.path.dirname(__file__), 'flask_session')
    os.makedirs(session_dir, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)