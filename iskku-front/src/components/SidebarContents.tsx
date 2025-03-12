import React, {useEffect, useState} from 'react';
import { Button, Divider, IconButton } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import SyncIcon from '@mui/icons-material/Sync';
import { useMessage } from '../utils/MessageContext';

const SidebarContents = () => {


    const { MessageState, MessageDispatch } = useMessage();
    const [ databaseUpdateTime, setDatabaseUpdateTime ] = useState('');
    const handleNewChat = () => {
        MessageDispatch({
            type: 'ADD_CHAT',
            chat_id: MessageState.ChatList.chatList.length + 1,
            message_list: { messages: [] },
        });
    };

    const handleChatChange = (chat_id: number) => {
        MessageDispatch({
            type: 'CHANGE_CHAT',
            chat_id: chat_id,
        });
    };

    // get database update time
    const getUpdateTime = async() => {
        
        try {
            const API_URL = process.env.REACT_APP_API_URL;
            if (!API_URL) {
                throw new Error('API_URL is not defined');
              }

            fetch(`${API_URL}/last_update`, {
                headers: {
                    "ngrok-skip-browser-warning": "69420",
                },
            }).then(response => {
                  if (!response.ok) {
                      throw new Error(`HTTP error! status: ${response.status}`);
                  }
                  return response.json();
              })
              .then(data => {
                setDatabaseUpdateTime(data.time);
              })
              .catch(error => {
                  console.error('Error fetching data:', error);
              });
            

        } catch (error) {
            console.error('Error getting database update time:', error);
        }
    };

    useEffect(() => {
        const fetchUpdateTime = async() => {
            await getUpdateTime();
        };
        fetchUpdateTime();
    }, []);
    
    return (
        <div style={{
            padding: '10%',
            display: 'flex',
            justifyContent: 'space-between',
            flexDirection: 'column',
            
        }}>
            <div style={{
                display: 'flex',
                justifyContent: 'center', // 가로 방향 가운데 정렬
                alignItems: 'center', // 세로 방향 가운데 정렬
                width: '100%', // 부모 요소의 너비를 100%로 설정
            }}>
                <img src='/assets/images/logo_grey.png' alt='logo' style={{width: '70%'}} />
            </div>
            <Button
                variant='text'
                startIcon={<AddIcon />}
                style={{
                    color: 'white',
                    fontSize: '1.2rem',
                    textTransform: 'none',
                    marginTop: '20%',
                    padding: '10px 20px',
                    justifyContent: 'start',
                    width: '100%',
                    fontFamily: 'BookkMyungjo-Bd',
                }}
                onClick={handleNewChat}
            >
                new chat
            </Button>
            <Divider style={{
                width: '100%', 
                background:'white'
            }} />
            <div
                style={{
                color: 'white',
                fontSize: '1.2rem',
                textTransform: 'none',
                marginTop: '20%',
                padding: '10px 20px',
                textAlign: 'left',
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                }}
            >
                <ChatBubbleOutlineIcon style={{ marginRight: '10px' }} />
                chat list
            </div>
            <Divider style={{
                width: '100%', 
                background:'white'
            }} />
            <div style={{
                width: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                marginTop: '20%',
                overflowY: 'auto',
            }}>
                {MessageState.ChatList.chatList.map((chat) => (
                    <Button
                        key={chat.chat_id}
                        variant='text'
                        style={{
                            color: 'white',
                            fontSize: '1.2rem',
                            textTransform: 'none',
                            padding: '10px 20px',
                            width: '100%',
                            fontFamily: 'BookkMyungjo-Bd',
                            justifyContent: 'start',
                        }}
                        onClick={() => handleChatChange(chat.chat_id)}
                    >
                        chat {chat.chat_id}
                    </Button>
                ))}
            </div>
            <div style={{
                display: 'flex', 
                flexDirection: 'row',
                alignItems: 'start',
            }}>
                <SyncIcon
                    style={{
                        color: 'white',
                        fontSize: '1.3rem',
                        textTransform: 'none',
                        marginTop: '20%',
                        textAlign: 'left',
                        width: '20%',
                    }}
                />
                <div style={{display: 'flex', flexDirection: 'column', width: '80%'}}>
                    <div style={{
                        color: 'white',
                        fontSize: '1.0rem',
                        textTransform: 'none',
                        marginTop: '20%',
                        textAlign: 'left',
                        width: '100%',
                    }}>
                        update database
                    </div>
                    <div style={{
                        color: 'white',
                        fontSize: '0.8rem',
                        textTransform: 'none',
                        textAlign: 'left',
                        width: '100%',
                    }}>
                        last updated<br />
                        {databaseUpdateTime}
                    </div>
                </div>
            </div>

        </div>
    );
}

export default SidebarContents;