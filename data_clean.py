import re
import argparse

def clean_whatsapp_chat(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        chat_content = f.read()

    # Remove timestamps
    chat_content = re.sub(r'\[\d+/\d+/\d+, \d+:\d+:\d+ [AP]M\] ', '', chat_content)

    # Remove template message
    chat_content = re.sub(r'Messages and calls are end-to-end encrypted\. No one outside of this chat, not even WhatsApp, can read or listen to them.', '', chat_content)
    
    chat_content = re.sub(r'((audio)|(video)|(image)) omitted', '' ,  chat_content)

    chat_content = re.sub(r'([A-Za-z0-9]| )+: \u200e\n', '', chat_content)

    chat_content = re.sub(r'\u200e', '', chat_content)


    print([c for c in chat_content[:250]])

    # Write the cleaned chat to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(chat_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean WhatsApp chat text file.')
    parser.add_argument('--input', type=str, help='Input file containing WhatsApp chat')
    parser.add_argument('--output', type=str, help='Output file to store cleaned WhatsApp chat')
    args = parser.parse_args()

    clean_whatsapp_chat(args.input, args.output)
