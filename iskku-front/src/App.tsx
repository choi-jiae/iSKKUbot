import React from 'react';

//import { useMediaQuery } from 'react-responsive';
import ChatbotContainer from './ChatbotContainer';
import Sidebar from 'components/Sidebar';
import Header from 'components/Header';

interface ItemType {
  isMobileSidebarOpen: boolean;
  onSidebarClose: (event: React.MouseEvent<HTMLElement>) => void;
  isSidebarOpen: boolean;
}

function App() {

  // const isMobile = useMediaQuery({
  //   query: '(max-width: 767px)'
  // });
  const [isSidebarOpen, setSidebarOpen] = React.useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = React.useState(false);

  return (
    <div 
      className = 'App'
      style = {{
        display: 'flex',
        flexDirection: 'row',
        width: '100%',
      }}
    >
      <Sidebar
        isMobileSidebarOpen={isMobileSidebarOpen}
        onSidebarClose={() => setIsMobileSidebarOpen(false)}
        isSidebarOpen={isSidebarOpen}
      />
      <div style={{
        display: 'flex', 
        flexDirection: 'column',
        width: '100%',
      }}>
      <Header toggleMobileSidebar={() => setIsMobileSidebarOpen(true)}/>
      <ChatbotContainer />
      </div>
    </div>
  );
}

export default App;
