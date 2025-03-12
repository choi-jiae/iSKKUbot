import React from 'react';
import { Box } from '@mui/material';

interface ChatBubbleProps {
    isChatbot: boolean;
    text: string;
};

const ChatBubble: React.FC<ChatBubbleProps> = ({isChatbot, text}) => {
    return (
        <div style={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'start',
            justifyContent: isChatbot ? 'flex-start' : 'flex-end',
            width: '100%',
            maxWidth: '100%',
            boxSizing: 'border-box',
        }}>
           {isChatbot && (
                <Box
                    sx={{
                        backgroundColor:'#f1f3f5',
                        borderRadius: 8,
                        padding: '1%',
                        margin: '1%',
                        width: 'fit-content',
                        display: 'flex',
                        justifyContent: 'center',
                    }}   
                >
                    <img 
                        src='/assets/images/logo_black.jpg' 
                        alt='chatbot' 
                        style={{
                            width: '50px',
                            borderRadius: '50%',
                            border: '1px solid #7C9D8A',
                        }}/>
                </Box>
            )}
            <Box
                sx={{
                    backgroundColor: isChatbot ? '#f1f3f5' : '#7C9D8A',
                    color: '#212121',
                    borderRadius: 5,
                    border: '1px solid #7C9D8A',
                    padding: '1% 2% 1% 2%',
                    margin: '1%',
                    maxWidth: '80%',
                    display: 'flex',
                    justifyContent: 'center',
                    flexDirection: 'column',
                    whiteSpace: 'pre-wrap',
                    boxSizing: 'border-box',
                    overflowWrap: 'break-word', 
                }}   
                dangerouslySetInnerHTML={{ __html: text }}
            />
                
        </div>
    );
};

export default ChatBubble;