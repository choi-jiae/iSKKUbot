import React from 'react';
import './css/ChatbotContainer.css';
import Header from './components/Header';
import MessageContainer from './MessageContainer';
import MessageInputField from 'components/MessageInputField';

const ChatbotContainer: React.FC = () => {
    
  
  return (

    <div className='chatbot-container'>
      <MessageContainer />
      <MessageInputField />
    </div>
  );
};

export default ChatbotContainer;