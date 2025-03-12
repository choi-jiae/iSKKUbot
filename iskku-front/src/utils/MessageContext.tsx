import React, { useReducer, useContext, createContext } from 'react';

interface Message {
    messages: Array<{
        message: string;
        isBot: boolean;
    }>;
}

interface ChatList {
    chatList: Array<
        {   
            chat_id: number;
            message_list: Message;
        }
    >;
}

type MessageState = {
    ChatList: ChatList,
    NowMessageList: Message,
    NowChatId: number,
};

type MessageAction = 
    | { type: 'ADD_MESSAGE'; message: string; isBot: boolean }
    | { type: 'UPDATE_MESSAGE'; message: string; isBot: boolean }
    | { type: 'ADD_CHAT'; chat_id: number; message_list: Message }
    | { type: 'CHANGE_CHAT'; chat_id: number;};

interface MessageContextType {
    MessageState: MessageState;
    MessageDispatch: React.Dispatch<MessageAction>;
}

interface MessageProviderProps {
    children: React.ReactNode;
}

const MessageStateContext = createContext<MessageContextType | undefined>(undefined);

const messageReducer = (state: MessageState, action: MessageAction): MessageState => {
    
    switch (action.type) {
        case 'ADD_MESSAGE':{
            return {
                ...state,
                NowMessageList: {
                    messages: [
                        ...state.NowMessageList.messages,
                        {
                            message: action.message,
                            isBot: action.isBot,
                        },
                    ],
                },
            };
        }
        case 'UPDATE_MESSAGE':{
            const updatedMessages = [...state.NowMessageList.messages];
            updatedMessages[updatedMessages.length - 1] = {
                ...updatedMessages[updatedMessages.length - 1],
                message: action.message,
            };
            return {
                ...state,
                NowMessageList: {
                    messages: updatedMessages,
                },

            };
        }
        case 'ADD_CHAT':{
            console.log(state.ChatList.chatList);
            const PreviousMessageList = [...state.NowMessageList.messages];
            const UpdateChatList  = [...state.ChatList.chatList];
            const PreviousChatId = state.NowChatId;

            UpdateChatList[PreviousChatId-1].message_list = {
                messages: PreviousMessageList,
            };

            return {
                ChatList: {
                    chatList: [
                        ...UpdateChatList,
                        {
                            chat_id: action.chat_id,
                            message_list: action.message_list,
                        },
                    ],
                },
                NowMessageList: action.message_list,
                NowChatId: action.chat_id,
            };
        }
        case 'CHANGE_CHAT':{
            console.log(action.chat_id);
            console.log(state.ChatList.chatList);
            const PreviousMessageList = [...state.NowMessageList.messages];
            const UpdateChatList  = [...state.ChatList.chatList];
            const PreviousChatId = state.NowChatId;

            UpdateChatList[PreviousChatId-1].message_list = {
                messages: PreviousMessageList,
            };

            const chat = state.ChatList.chatList.find((chat) => chat.chat_id === action.chat_id);
            if (chat) {
                return {
                    ChatList: {
                        chatList:UpdateChatList,
                    },
                    NowMessageList: chat.message_list,
                    NowChatId: action.chat_id,
                    
                };
            }
            return state;
        }
        default:
            return state;
    }
};

export const MessageProvider: React.FC<MessageProviderProps> = ({ children }) => {
    const [MessageState, MessageDispatch] = useReducer(messageReducer, {
        ChatList: { chatList: [{ chat_id: 1, message_list: { messages: [] } }] },
        NowMessageList: { messages: [] },
        NowChatId: 1,
    });

    return (
        <MessageStateContext.Provider value={{ MessageState, MessageDispatch }}>
            {children}
        </MessageStateContext.Provider>
    );
};

export const useMessage = () => {
    const state = useContext(MessageStateContext);
    if (!state) {
        throw new Error('Cannot find MessageProvider');
    }
    return state;
}