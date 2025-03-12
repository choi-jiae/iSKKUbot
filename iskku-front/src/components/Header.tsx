import React from 'react';
import '../css/Header.css';
import { useMediaQuery } from 'react-responsive';
import { IconButton } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';

interface HeaderProps {
  toggleMobileSidebar: (event: React.MouseEvent<HTMLElement>) => void;
}

const Header = ({toggleMobileSidebar}: HeaderProps) => {

  const isMobile = useMediaQuery({
    query: '(max-width: 767px)'
  });

  const version = '1.0.0';


  if (isMobile) {
    return (
      <header className='header'>
        <IconButton 
          size='large'
          onClick={toggleMobileSidebar}
        >
          <MenuIcon />
        </IconButton>
        <div style={{display:'flex', flexDirection: 'column'}}>
          <h1>iSKKUbot</h1>
          <h4>version {version}</h4>
        </div>
      </header>
    );
  }
  return (
    <header className='header'>
      <div style={{display:'flex', flexDirection: 'column'}}>
        <h1>iSKKUbot</h1>
        <h4>version {version}</h4>
      </div>
    </header>
  );
};

export default Header;