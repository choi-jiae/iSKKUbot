import React, { useState } from 'react';
import '../css/MessageInputField.css';
import { Box, IconButton, InputBase } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import { useMessage } from '../utils/MessageContext';
import StopCircleIcon from '@mui/icons-material/StopCircle';

const MessageInputField: React.FC = () => {
    const { MessageState, MessageDispatch } = useMessage();
    const [ isChatbotTyping,  setIsChatbotTyping ] = useState(false); 

    const handleSendMessage = async() => {
      const inputMessage = (document.getElementById('message-input') as HTMLInputElement)?.value;
      MessageDispatch({
        type: 'ADD_MESSAGE',
        message: inputMessage,
        isBot: false,
      });

      (document.getElementById('message-input') as HTMLInputElement).value = '';
      setIsChatbotTyping(true);
      // TODO: send message to backend
      try {
        MessageDispatch({
          type: 'ADD_MESSAGE',
          message: '...loading',
          isBot: true,
        });
        const API_URL = process.env.REACT_APP_API_URL;
        const chatbotResponse = await fetch(API_URL+'/chat',{
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            chat: inputMessage
          }),
        });

        if (!chatbotResponse.ok){
          throw new Error('Chatbot response is not ok');
        }

        // response ok
        const reader = chatbotResponse.body?.getReader();
        const decoder = new TextDecoder();
        let textResult = '';

        while (true){
          const result = await reader?.read();

          if (!result) break;

          const { done, value } = result;

          if (done){
            setIsChatbotTyping(false);
            break;
          }
          const text = decoder.decode(value, {stream: true});
          textResult += text.replace(/\n/g, "<br>");

          MessageDispatch({
            type: 'UPDATE_MESSAGE',
            message: textResult,
            isBot: true,
          });
        }
      } catch (error) {
        setIsChatbotTyping(false);
        console.error('Failed to send message to backend', error);
        MessageDispatch({
          type: 'UPDATE_MESSAGE',
          message: 'Failed to receive response from chatbot',
          isBot: true,
        });
      }
    }

    return (
        <div className = 'message-input-field'>
          <Box
            sx={{
              width: '80%',
              borderRadius: 8,
              backgroundColor: '#7C9D8A',
              display: 'flex',
              padding: '0.3% 1% 0.3% 3%',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
            
          >
            <InputBase
                id='message-input'
                multiline={true}
                maxRows={4}
                placeholder = 'please enter your question'
                color='secondary'
                sx={{
                  marginRight: '1%',
                  width: '90%',
                  fontFamily: 'BookkMyungjo-Bd',

                }}
                onKeyDown={(event) => {
                  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
                  if (!isMobile && !isChatbotTyping && event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    handleSendMessage();
                  }
                }}
            />
            <IconButton
              sx={{
                backgroundColor: '#f1f3f5', // 버튼 배경색
                color: 'black.700',
                ":hover": {
                  backgroundColor: '#dee2e6', // 마우스 오버시 배경색
                },
                borderRadius: '50%', // 동그란 버튼
                padding: '1px',
                marginTop: '1%',
                marginBottom: '1%',
                size: 'fit-content',
              }}
              disabled={isChatbotTyping}
              onClick={handleSendMessage}
            >
              {isChatbotTyping ? 
                <StopCircleIcon
                  sx ={{
                    fontWeight: 'bold',
                    size: '20px',
                  }}
                />:
                <ArrowUpwardIcon 
                  sx ={{
                    fontWeight: 'bold',
                    size: '20px',
                  }}/>
                }
            </IconButton>

          
          </Box>

            
        </div>
    );
    }

export default MessageInputField;