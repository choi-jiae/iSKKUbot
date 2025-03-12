import React from 'react';
import ChatBubble from 'components/ChatBubble';
import './css/MessageContainer.css';
import { useMessage } from './utils/MessageContext';


const MessageContainer: React.FC = () => {


  const { MessageState, MessageDispatch } = useMessage();
 
  return (
    <div className='message-container'>
      {
        MessageState.NowMessageList.messages.map((message, index) => {
          return <ChatBubble key={index} isChatbot= {message.isBot} text={message.message}/>;
        })
      }
    </div>
  );
}

export default MessageContainer;