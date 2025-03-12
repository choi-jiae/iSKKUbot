import { Box, Drawer } from "@mui/material";
import { useMediaQuery } from 'react-responsive';
import React from "react";
import SidebarContents from "./SidebarContents";

interface ItemType {
  isMobileSidebarOpen: boolean;
  onSidebarClose: (event: React.MouseEvent<HTMLElement>) => void;
  isSidebarOpen: boolean;
}

const Sidebar = ({
  isMobileSidebarOpen,
  onSidebarClose,
  isSidebarOpen,
}: ItemType) => {
    const isMobile = useMediaQuery({
      query: '(max-width: 767px)'
    });

  const sidebarWidth = "270px";

  if (isMobile) {
    return (
      <Drawer
        anchor="left"
        open={isMobileSidebarOpen}
        onClose={onSidebarClose}
        variant="temporary"
        PaperProps={{
          sx: {
            width: sidebarWidth,
            boxShadow: (theme) => theme.shadows[8],
            backgroundColor: '#003225',
          },
        }}
      >

        <Box px={2}>
          <SidebarContents />
        </Box>

        
      </Drawer>
    );
  }

  return (
    <Box
      sx={{
        width: sidebarWidth,
        flexShrink: 0,
      }}
    >
      {/* ------------------------------------------- */}
      {/* Sidebar for desktop */}
      {/* ------------------------------------------- */}
      <Drawer
        anchor="left"
        open={isSidebarOpen}
        variant="permanent"
        PaperProps={{
          sx: {
            width: sidebarWidth,
            boxSizing: "border-box",
            backgroundColor: '#003225',
          },
        }}
      >

        <Box
          sx={{
            height: "100%",
            color: '#003225',
          }}
        >
          <SidebarContents />
        </Box>
      </Drawer>
    </Box>
  );


};

export default Sidebar;