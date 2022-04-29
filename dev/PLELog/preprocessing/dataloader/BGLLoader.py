import sys

sys.path.extend([".",".."])
from CONSTANTS import *
from collections import OrderedDict
from preprocessing.BasicLoader import BasicDataLoader


class BGLLoader(BasicDataLoader):
    def __init__(self, in_file=None,
                 window_size=120,
                 dataset_base=os.path.join(PROJECT_ROOT, 'datasets/BGL'),
                 semantic_repr_func=None):
        super(BGLLoader, self).__init__()

        # Construct logger.
        self.logger = logging.getLogger('BGLLoader')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'BGLLoader.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.info(
            'Construct self.logger success, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))

        if not os.path.exists(in_file):
            self.logger.error('Input file not found, please check.')
            exit(1)
        self.in_file = in_file
        self.remove_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.window_size = window_size
        self.dataset_base = dataset_base
        self._load_raw_log_seqs()
        self.semantic_repr_func = semantic_repr_func
        pass

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for id, token in enumerate(tokens):
            if id not in self.remove_cols:
                after_process.append(token)
        return ' '.join(after_process)
        # return re.sub('[\*\.\?\+\$\^\[\]\(\)\{\}\|\\\/]', '', ' '.join(after_process))

    def parse_by_Official(self):
        self._restore()
        # Define official templates
        templates = ["(.*):(.*) (.*):(.*) (.*):(.*) (.*):(.*)",
                     "(.*) (.*) (.*) BGLERR_IDO_PKT_TIMEOUT connection lost to nodelinkservice card",
                     "(.*) correctable errors exceeds threshold iar (.*) lr (.*)",
                     "(.*) ddr errors detected and corrected on rank (.*) symbol (.*) over (.*) seconds",
                     "(.*) ddr errorss detected and corrected on rank (.*) symbol (.*), bit (.*)",
                     "(.*) double-hummer alignment exceptions",
                     "(.*) exited abnormally due to signal: Aborted",
                     "(.*) exited normally with exit code (.*)",
                     "(.*) floating point alignment exceptions",
                     "(.*) L3 (.*) errors dcr (.*) detected and corrected over (.*) seconds",
                     "(.*) L3 (.*) errors dcr (.*) detected and corrected",
                     "(.*) microseconds spent in the rbs signal handler during (.*) calls (.*) microseconds was the maximum time for a single instance of a correctable ddr",
                     "(.*) PGOOD error latched on link card",
                     "(.*) power module (.*) is not accessible",
                     "(.*) TLB error interrupt",
                     "(.*) torus non-crc errors dcr (.*) detected and corrected over (.*) seconds",
                     "(.*) torus non-crc errors dcr (.*) detected and corrected",
                     "(.*) torus receiver (.*) input pipe errors dcr (.*) detected and corrected",
                     "(.*) torus receiver (.*) input pipe errors dcr (.*) detected and corrected over (.*) seconds",
                     "(.*) torus (.*) (.*) (.*) errors dcr (.*) detected and corrected",
                     "(.*) torus (.*) (.*) (.*) errors dcr (.*) detected and corrected over (.*) seconds",
                     "(.*) total interrupts (.*) critical input interrupts (.*) microseconds total spent on critical input interrupts, (.*) microseconds max time in a critical input interrupt",
                     "(.*) tree receiver (.*) in re-synch state events dcr (.*) detected",
                     "(.*) tree receiver (.*) in re-synch state events dcr (.*) detected over (.*) seconds",
                     "Added (.*) subnets and (.*) addresses to DB",
                     "address parity error0",
                     "auxiliary processor0",
                     "Bad cable going into LinkCard (.*) Jtag (.*) Port (.*) - (.*) bad wires",
                     "BglIdoChip table has (.*) IDOs with the same IP address (.*)",
                     "BGLMASTER FAILURE mmcs_server exited normally with exit code 13",
                     "BGLMaster has been started: BGLMaster --consoleip 127001 --consoleport 32035 --configfile bglmasterinit",
                     "BGLMaster has been started: BGLMaster --consoleip 127001 --consoleport 32035 --configfile bglmasterinit --autorestart y",
                     "BGLMaster has been started: BGLMaster --consoleip 127001 --consoleport 32035 --configfile bglmasterinit --autorestart y --db2profile ubgdb2cli",
                     "byte ordering exception0",
                     "Can not get assembly information for node card",
                     "capture (.*)",
                     "capture first (.*) (.*) error address0",
                     "CE sym (.*) at (.*) mask (.*)",
                     "CHECK_INITIAL_GLOBAL_INTERRUPT_VALUES",
                     "chip select0",
                     "ciod: (.*) coordinate (.*) exceeds physical dimension (.*) at line (.*) of node map file (.*)",
                     "ciod: cpu (.*) at treeaddr (.*) sent unrecognized message (.*)",
                     "ciod: duplicate canonical-rank (.*) to logical-rank (.*) mapping at line (.*) of node map file (.*)",
                     "ciod: Error creating node map from file (.*) Argument list too long",
                     "ciod: Error creating node map from file (.*) Bad address",
                     "ciod: Error creating node map from file (.*) Bad file descriptor",
                     "ciod: Error creating node map from file (.*) Block device required",
                     "ciod: Error creating node map from file (.*) Cannot allocate memory",
                     "ciod: Error creating node map from file (.*) Device or resource busy",
                     "ciod: Error creating node map from file (.*) No child processes",
                     "ciod: Error creating node map from file (.*) No such file or directory",
                     "ciod: Error creating node map from file (.*) Permission denied",
                     "ciod: Error creating node map from file (.*) Resource temporarily unavailable",
                     "ciod: Error loading (.*) invalid or missing program image, Exec format error",
                     "ciod: Error loading (.*) invalid or missing program image, No such device",
                     "ciod: Error loading (.*) invalid or missing program image, No such file or directory",
                     "ciod: Error loading (.*) invalid or missing program image, Permission denied",
                     "ciod: Error loading (.*) not a CNK program image",
                     "ciod: Error loading (.*) program image too big, (.*) > (.*)",
                     "ciod: Error loading -mode VN: invalid or missing program image, No such file or directory",
                     "ciod: Error opening node map file (.*) No such file or directory",
                     "ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to (.*) (.*) (.*) (.*) (.*)",
                     "ciod: Error reading message prefix on CioStream socket to (.*) Connection reset by peer",
                     "ciod: Error reading message prefix on CioStream socket to (.*) Connection timed out",
                     "ciod: Error reading message prefix on CioStream socket to (.*) Link has been severed",
                     "ciod: failed to read message prefix on control stream CioStream socket to (.*)",
                     "ciod: for node (.*) incomplete data written to core file (.*)",
                     "ciod: for node (.*) read continuation request but ioState is (.*)",
                     "ciod: generated (.*) core files for program (.*)",
                     "ciod: In packet from node (.*) (.*) message code (.*) is not (.*) or 4294967295 (.*) (.*) (.*) (.*)",
                     "ciod: In packet from node (.*) (.*) message still ready for node (.*) (.*) (.*) (.*) (.*)",
                     "ciod: LOGIN chdir(.*) failed: Inputoutput error",
                     "ciod: LOGIN chdir(.*) failed: No such file or directory",
                     "ciod: LOGIN (.*) failed: Permission denied",
                     "ciod: Message code (.*) is not (.*) or 4294967295",
                     "ciod: Missing or invalid fields on line (.*) of node map file (.*)",
                     "ciod: pollControlDescriptors: Detected the debugger died",
                     "ciod: Received signal (.*) (.*) (.*) (.*)",
                     "ciod: sendMsgToDebugger: error sending PROGRAM_EXITED message to debugger",
                     "ciod: Unexpected eof at line (.*) of node map file (.*)",
                     "ciodb has been restarted",
                     "close EDRAM pages as soon as possible0",
                     "comibmbgldevicesBulkPowerModule with VPD of comibmbgldevicesBulkPowerModuleVpdReply: IBM Part Number: 53P5763, Vendor: Cherokee International, Vendor Serial Number: 4274124, Assembly Revision:",
                     "command manager unit summary0",
                     "Controlling BGL rows  (.*)",
                     "core configuration register: (.*)",
                     "Core Configuration Register 0: (.*)",
                     "correctable (.*)",
                     "correctable error detected in directory (.*)",
                     "correctable error detected in EDRAM bank (.*)",
                     "critical input interrupt (.*)",
                     "critical input interrupt (.*) (.*) warning for (.*) (.*) wire",
                     "critical input interrupt (.*) (.*) warning for torus (.*) wire, suppressing further interrupts of same type",
                     "data (.*) plb (.*)",
                     "data address: (.*)",
                     "data address space0",
                     "data cache (.*) parity error detected attempting to correct",
                     "data storage interrupt",
                     "data store interrupt caused by (.*)",
                     "data TLB error interrupt data address space0",
                     "dbcr0=(.*) dbsr=(.*) ccr0=(.*)",
                     "d-cache (.*) parity (.*)",
                     "DCR (.*) : (.*)",
                     "ddr: activating redundant bit steering for next allocation: (.*) (.*)",
                     "ddr: activating redundant bit steering: (.*) (.*)",
                     "ddr: excessive soft failures, consider replacing the card",
                     "DDR failing (.*) register: (.*) (.*)",
                     "DDR failing info register: (.*)",
                     "DDR failing info register: DDR Fail Info Register: (.*)",
                     "DDR machine check register: (.*) (.*)",
                     "ddr: redundant bit steering failed, sequencer timeout",
                     "ddr: Suppressing further CE interrupts",
                     "ddr: Unable to steer (.*) (.*) - rank is already steering symbol (.*) Due to multiple symbols being over the correctabl[e]{0,1}",
                     "ddr: Unable to steer (.*) (.*) - rank is already steering symbol (.*) Due to multiple symbols being over the correctable e",
                     "ddr: Unable to steer (.*) (.*) - rank is already steering symbol (.*) Due to multiple symbols being over the correctable error threshold, consider replacing the card",
                     '(.*) error threshold, consider replacing the card',
                     "ddrSize == (.*)  ddrSize == (.*)",
                     "debug interrupt enable0",
                     "debug wait enable0",
                     "DeclareServiceNetworkCharacteristics has been run but the DB is not empty",
                     "DeclareServiceNetworkCharacteristics has been run with the force option but the DB is not empty",
                     "disable all access to cache directory0",
                     "disable apu instruction broadcast0",
                     "disable flagging of DDR UE's as major internal error0",
                     "disable speculative access0",
                     "disable store gathering0",
                     "disable trace broadcast0",
                     "disable write lines 2:40",
                     "divide-by-zero (.*)",
                     "enable (.*) exceptions0",
                     "enable invalid operation exceptions0",
                     "enable non-IEEE mode0",
                     "enabled exception summary0",
                     "EndServiceAction (.*) performed upon (.*) by (.*)",
                     "EndServiceAction (.*) was performed upon (.*) by (.*)",
                     "EndServiceAction is restarting the (.*) cards in Midplane (.*) as part of Service Action (.*)",
                     "EndServiceAction is restarting the (.*) in midplane (.*) as part of Service Action (.*)",
                     "Error getting detailed hw info for node, caught javaioIOException: Problems with the chip, clear all resets",
                     "Error getting detailed hw info for node, caught javaioIOException: Problems with the chip, could not enable clock domains",
                     "Error getting detailed hw info for node, caught javaioIOException: Problems with the chip, could not pull all resets",
                     "Error receiving packet on tree network, expecting type (.*) instead of type (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "Error receiving packet on tree network, packet index (.*) greater than max 366 (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "Error sending packet on tree network, packet at address (.*) is not aligned",
                     "error threshold, consider replacing the card",
                     "Error: unable to mount filesystem",
                     "exception (.*)",
                     "Exception Syndrome Register: (.*)",
                     "exception syndrome register: (.*)",
                     "Expected 10 active FanModules, but found 9  Found (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "external input interrupt (.*)",
                     "external input interrupt (.*) (.*) (.*) tree receiver (.*) in resynch mode",
                     "external input interrupt (.*) (.*) number of corrected SRAM errors has exceeded threshold",
                     "external input interrupt (.*) (.*) number of corrected SRAM errors has exceeded threshold, suppressing further interrupts of same type",
                     "external input interrupt (.*) (.*) torus sender (.*) retransmission error was corrected",
                     "external input interrupt (.*) (.*) tree header with no target waiting",
                     "external input interrupt (.*) (.*) uncorrectable torus error",
                     "floating point (.*)",
                     "floating point instr (.*)",
                     "Floating Point Registers:",
                     "Floating Point Status and Control Register: (.*)",
                     "floating point unavailable interrupt",
                     "floating pt ex mode (.*) (.*)",
                     "force loadstore alignment0",
                     "Found invalid node ecid in processor card slot (.*) ecid 0000000000000000000000000000",
                     "fpr(.*)=(.*) (.*) (.*) (.*)",
                     "fraction (.*)",
                     "General Purpose Registers:",
                     "general purpose registers:",
                     "generating (.*)",
                     "gister: machine state register: machine state register: machine state register: machine state register: machine state register:",
                     "guaranteed (.*) cache block (.*)",
                     "Hardware monitor caught javalangIllegalStateException: while executing I2C Operation caught javanetSocketException: Broken pipe and is stopping",
                     "Hardware monitor caught javanetSocketException: Broken pipe and is stopping",
                     "iar (.*) dear (.*)",
                     "i-cache parity error0",
                     "icache prefetch (.*)",
                     "Ido chip status changed: (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "Ido packet timeout",
                     "idoproxy communication failure: socket closed",
                     "idoproxydb has been started: Name: (.*)  Input parameters: -enableflush -loguserinfo dbproperties BlueGene1",
                     "idoproxydb hit ASSERT condition: ASSERT expression=(.*) Source file=(.*) Source line=(.*) Function=(.*) IdoTransportMgr::SendPacketIdoUdpMgr(.*), BglCtlPavTrace(.*)",
                     "idoproxydb hit ASSERT condition: ASSERT expression=!nMsgLen > 0x10000 Source file=idomarshaleriocpp Source line=1929 Function=int IdoMarshalerRecvBuffer::ReadBlockIdoMsg::IdoMsgHdr(.*)&",
                     "idoproxydb hit ASSERT condition: ASSERT expression=pTargetMgr Source file=idoclientmgrcpp Source line=353 Function=int IdoClientMgr::TargetCloseconst char(.*)",
                     "idoproxydb hit ASSERT condition: ASSERT expression=!RecvMsgHdrulLen > 0x10000 Source file=idomarshaleriocpp Source line=387 Function=virtual int IdoMarshalerIo::RunRecv",
                     "imprecise machine (.*)",
                     "inexact (.*)",
                     "0x[0-9a-fA-F]+ 0x[0-9a-fA-F]+",
                     "instance of a correctable ddr RAS KERNEL INFO (.*) microseconds spent in the rbs signal handler during (.*) calls (.*) microseconds was the maximum time for a single instance of a correctable ddr",
                     "instruction address: (.*)",
                     "instruction address space0",
                     "instruction cache parity error corrected",
                     "instruction plb (.*)",
                     "interrupt threshold0",
                     "invalid (.*)",
                     "invalid operation exception (.*)",
                     "job (.*) timed out Block freed",
                     "Kernel detected (.*) integer alignment exceptions (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*)",
                     "kernel panic",
                     "L1 DCACHE summary averages: #ofDirtyLines: (.*) out of 1024 #ofDirtyDblWord: (.*) out of 4096",
                     "L3 (.*) (.*) register: (.*)",
                     "L3 major internal error",
                     "LinkCard is not fully functional",
                     "lr:(.*) cr:(.*) xer:(.*) ctr:(.*)",
                     "Lustre mount FAILED : (.*) : block_id : location",
                     "Lustre mount FAILED : (.*) : point pgb1",
                     "machine check (.*)",
                     "MACHINE CHECK DCR read timeout (.*) iar (.*) lr (.*)",
                     "machine check: i-fetch0",
                     "machine check interrupt (.*) L2 dcache unit (.*) (.*) parity error",
                     "machine check interrupt (.*) L2 DCU read error",
                     "machine check interrupt (.*) L3 major internal error",
                     "machine check interrupt (.*) TorusTreeGI read error 0",
                     "machine check interrupt (.*) L2 dcache unit data parity error",
                     "MACHINE CHECK PLB write IRQ (.*) iar (.*) lr (.*)",
                     "Machine Check Status Register: (.*)",
                     "machine check status register: (.*)",
                     "machine state register:",
                     "Machine State Register: (.*)",
                     "machine state register: (.*)",
                     "machine state register: machine state register: machine state register: machine state register: machine state register: machine",
                     "MailboxMonitor::serviceMailboxes lib_ido_error: -1019 socket closed",
                     "MailboxMonitor::serviceMailboxes lib_ido_error: -1114 unexpected socket error: Broken pipe",
                     "mask(.*)",
                     "max number of outstanding prefetches7",
                     "max time in a cr RAS KERNEL INFO (.*) total interrupts (.*) critical input interrupts (.*) microseconds total spent on critical input interrupts, (.*) microseconds max time in a critical input interrupt",
                     "memory and bus summary0",
                     "memory manager (.*)",
                     "memory manager  command manager address parity0",
                     "memory manager address error0",
                     "memory manager address parity error0",
                     "memory manager refresh contention0",
                     "memory manager refresh counter timeout0",
                     "memory manager RMW buffer parity0",
                     "memory manager store buffer parity0",
                     "memory manager strobe gate0",
                     "memory manager uncorrectable (.*)",
                     "Microloader Assertion",
                     "MidplaneSwitchController performing bit sparing on (.*) bit (.*)",
                     "MidplaneSwitchController::clearPort bll_clear_port failed: (.*)",
                     "MidplaneSwitchController::parityAlignment pap failed: (.*) (.*) (.*)",
                     "MidplaneSwitchController::receiveTrain iap failed: (.*) (.*) (.*)",
                     "MidplaneSwitchController::sendTrain port disconnected: (.*)",
                     "minus (.*) (.*)",
                     "minus (.*)",
                     "miscompare0",
                     "Missing reverse cable: Cable (.*) (.*) (.*) (.*) --> (.*) (.*) (.*) (.*) is present BUT the reverse cable (.*) (.*) (.*) (.*) --> (.*) (.*) (.*) (.*) is missing",
                     "mmcs_db_server has been started: bglBlueLightppcfloorbglsysbinmmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
                     "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties dbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
                     "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties dbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --shutdown-timeout 120",
                     "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties dbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --shutdown-timeout 120 --shutdown-timeout 240",
                     "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
                     "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
                     "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --no-reconnect-blocks",
                     "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --shutdown-timeout (.*)",
                     "mmcs_server exited abnormally due to signal: Segmentation fault",
                     "monitor caught javalangIllegalStateException: while executing CONTROL Operation caught javaioEOFException and is stopping",
                     "monitor caught javalangIllegalStateException: while executing (.*) Operation caught javanetSocketException: Broken pipe and is stopping",
                     "monitor caught javalangUnsupportedOperationException: power module (.*) not present and is stopping",
                     "msr=(.*) dear=(.*) esr=(.*) fpscr=(.*)",
                     "New ido chip inserted into the database: (.*) (.*) (.*) (.*)",
                     "NFS Mount failed on (.*) slept (.*) seconds, retrying (.*)",
                     "no ethernet link",
                     "No power module (.*) found found on link card",
                     "Node card is not fully functional",
                     "Node card status: ALERT 0, ALERT 1, ALERT 2, ALERT 3 is are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted TEMPERATURE MASK IS ACTIVE No temperature error Temperature Limit Error Latch is clear PGOOD is asserted PGOOD error latch is clear MPGOOD is OK MPGOOD error latch is clear The 25 volt rail is OK The 15 volt rail is OK",
                     "Node card status: ALERT 0, ALERT 1, ALERT 2, ALERT 3 is are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD is asserted PGOOD error latch is clear MPGOOD is OK MPGOOD error latch is clear The 25 volt rail is OK The 15 volt rail is OK",
                     "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD IS NOT OK MPGOOD ERROR LATCH IS ACTIVE THE 25 VOLT RAIL IS NOT OK THE 15 VOLT RAIL IS NOT OK",
                     "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD IS NOT OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
                     "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD is OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
                     "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD is OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
                     "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD IS NOT OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
                     "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD is OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
                     "Node card VPD check: (.*) node in processor card slot (.*) do not match VPD ecid (.*) found (.*)",
                     "Node card VPD check: missing (.*) node, VPD ecid (.*) in processor card slot (.*)",
                     "NodeCard is not fully functional",
                     "NodeCard temperature sensor chip (.*) is not accessible",
                     "NodeCard VPD chip is not accessible",
                     "NodeCard VPD is corrupt",
                     "number of correctable errors detected in L3 (.*)",
                     "number of lines with parity errors written to L3 (.*)",
                     "overflow exception0",
                     "parity error in bank (.*)",
                     "parity error in read queue (.*)",
                     "parity error in write buffer0",
                     "parity error0",
                     "plus (.*)",
                     "Power deactivated: (.*)",
                     "Power Good signal deactivated: (.*) A service action may be required",
                     "power module status fault detected on node card status registers are: (.*)",
                     "prefetch depth for core (.*)",
                     "prefetch depth for PLB slave1",
                     "PrepareForService is being done on this (.*) (.*) (.*) (.*) (.*) by (.*)",
                     "PrepareForService is being done on this Midplane (.*) (.*) (.*) by (.*)",
                     "PrepareForService is being done on this part (.*) (.*) (.*) (.*) (.*) by (.*)",
                     "PrepareForService is being done on this rack (.*) by (.*)",
                     "PrepareForService shutting down (.*) as part of Service Action (.*)",
                     "Problem communicating with link card iDo machine with LP of (.*) caught javalangIllegalStateException: while executing I2C Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 13 and card type 1 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = 845, Reply Sequence Number = -1, timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = -1, Expected Sync Command = 32, Actual Sync Reply = -1",
                     "Problem communicating with node card, iDo machine with LP of (.*) caught javalangIllegalStateException: while executing (.*) Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 13 and card type 4 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = 0, Reply Sequence Number = -1, timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = -1, Expected Sync Command = 8, Actual Sync Reply = -1",
                     "Problem communicating with service card, ido chip: (.*) javaioIOException: Could not find EthernetSwitch on port:address (.*)",
                     "Problem communicating with service card, ido chip: (.*) javalangIllegalStateException: IDo is not in functional state -- currently in state COMMUNICATION_ERROR",
                     "Problem communicating with service card, ido chip: (.*) javalangIllegalStateException: while executing CONTROL Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 9 and card type 2 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = (.*) Reply Sequence Number = (.*) timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = (.*) Expected Sync Command = 8, Actual Sync Reply = (.*)",
                     "Problem reading the ethernet arl entries fro the service card: javalangIllegalStateException: while executing I2C Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 9 and card type 2 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = (.*) Reply Sequence Number = (.*) timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = (.*) Expected Sync Command = 32, Actual Sync Reply = (.*)",
                     "problem state (.*)",
                     "program interrupt",
                     "program interrupt: fp compare0",
                     "program interrupt: fp cr (.*)",
                     "program interrupt: illegal (.*)",
                     "program interrupt: imprecise exception0",
                     "program interrupt: privileged instruction0",
                     "program interrupt: trap (.*)",
                     "program interrupt: unimplemented operation0",
                     "quiet NaN0",
                     "qw trapped0",
                     "r(.*)=(.*) r(.*)=(.*) r(.*)=(.*) r(.*)=(.*)",
                     "regctl scancom interface0",
                     "reserved0",
                     "round nearest0",
                     "round toward (.*)",
                     "rts assertion failed: `personality->version == BGLPERSONALITY_VERSION' in `void start' at startcc:131",
                     "rts assertion failed: `vaddr % PAGE_SIZE_1M == 0' in `int initializeAppMemoryint, TLBEntry&, unsigned int, unsigned int' at mmucc:540",
                     "rts: bad message header: cpu (.*) invalid (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "rts: bad message header: expecting type (.*) instead of type (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "rts: bad message header: index 0 greater than total 0 (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "rts: bad message header: packet index (.*) greater than max 366 (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "rts internal error",
                     "rts: kernel terminated for reason (.*)",
                     "rts: kernel terminated for reason 1001rts: bad message header: invalid cpu, (.*) (.*) (.*) (.*)",
                     "rts: kernel terminated for reason 1002rts: bad message header: too many packets, (.*) (.*) (.*) (.*) (.*)",
                     "rts: kernel terminated for reason 1004rts: bad message header: expecting type (.*) (.*) (.*) (.*) (.*)",
                     "rts panic! - stopping execution",
                     "rts treetorus link training failed: wanted: (.*) got: (.*)",
                     "Running as background command",
                     "(.*) (.*) StatusA",
                     "shutdown complete",
                     "size of DDR we are caching1 512M",
                     "size of scratchpad portion of L30 0M",
                     "Special Purpose Registers:",
                     "special purpose registers:",
                     "start (.*)",
                     "Starting SystemController",
                     "state machine0",
                     "state register: machine state register: machine state register: machine state register: machine state register: machine state re",
                     "store (.*)",
                     "summary1",
                     "suppressing further interrupts of same type",
                     "symbol(.*)",
                     "Target=(.*) Message=(.*)",
                     "(.*) (.*) All all zeros, power good may be low",
                     "(.*) (.*) failed to lock",
                     "(.*) (.*) JtagId = (.*)",
                     "(.*) (.*) JtagId = (.*) Run environmental monitor to diagnose possible hardware failure",
                     "Temperature Over Limit on link card",
                     "this link card is not fully functional",
                     "tlb (.*)",
                     "Torus non-recoverable error DCRs follow",
                     "total of (.*) ddr errors detected and corrected",
                     "total of (.*) ddr errors detected and corrected over (.*) seconds",
                     "turn on hidden refreshes1",
                     "uncorrectable (.*)",
                     "uncorrectable error detected in (.*) (.*)",
                     "uncorrectable error detected in EDRAM bank (.*)",
                     "underflow (.*)",
                     "VALIDATE_LOAD_IMAGE_CRC_IN_DRAM",
                     "wait state enable0",
                     "While initializing link card iDo machine with LP of (.*) caught javaioIOException: Could not contact iDo with (.*) and (.*) because javalangRuntimeException: Communication error: (.*)",
                     "While initializing node card, ido with LP of (.*) caught javalangIllegalStateException: IDo is not in functional state -- currently in state COMMUNICATION_ERROR",
                     "While initializing node card, ido with LP of (.*) caught javalangIllegalStateException: while executing CONTROL Operation caught javalangRuntimeException: Communication error: (.*)",
                     "While initializing node card, ido with LP of (.*) caught javalangNullPointerException",
                     "While initializing node card, ido with LP of (.*) caught javalangIllegalStateException: while executing I2C Operation caught javalangRuntimeException: Communication error: (.*)",
                     "While initializing (.*) card iDo with LP of (.*) caught javaioIOException: Could not contact iDo with LP=(.*) and IP=(.*) because javalangRuntimeException: Communication error: DirectIDo for Uninitialized DirectIDo for (.*) is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = 0, Reply Sequence Number = -1, timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 5, Actual Op Reply = -1, Expected Sync Command = 10, Actual Sync Reply = -1",
                     "While inserting monitor info into DB caught javalangNullPointerException",
                     "While reading FanModule caught javalangIllegalStateException: while executing I2C Operation caught (.*)",
                     "While reading FanModule caught javalangIllegalStateException: while executing I2C Operation caught javanetSocketException: Broken pipe",
                     "While setting fan speed caught javalangIllegalStateException: while executing I2C Operation caught (.*)",
                     "write buffer commit threshold2",
                     "DDR failing data registers: (.*) (.*)",
                     "program interrupt: fp cr field 0",
                     "rts treetorus link training failed: wanted: (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) got: (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
                     "rts: kernel terminated for reason 1002rts: bad message header: too many packets, (.*) (.*) (.*) (.*)",
                     '(.*)'
                     ]

        save_path = os.path.join(PROJECT_ROOT, 'datasets/BGL/persistences/official')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        templates_file = os.path.join(save_path, 'NC_templates.txt')
        log2temp_file = os.path.join(save_path, 'log2temp.txt')
        if os.path.exists(templates_file) and os.path.exists(log2temp_file):
            self.logger.info('Found parsing result, please note that this does not guarantee a smooth execution.')
            with open(templates_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(',')
                    id = int(tokens[0])
                    template = ','.join(tokens[1:])
                    self.templates[id] = template

            with open(log2temp_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    logid, tempid = line.strip().split(',')
                    self.log2temp[int(logid)] = int(tempid)

            pass

        else:
            for id, template in enumerate(templates):
                self.templates[id] = template
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    line = line.strip()
                    if self.remove_cols:
                        processed_line = self._pre_process(line)
                    for index, template in self.templates.items():
                        if re.compile(template).match(processed_line) is not None:
                            self.log2temp[log_id] = index
                            break
                    if log_id not in self.log2temp.keys():
                        # if processed_line == '':
                        #     self.log2temp[log_id] = -1
                        self.logger.warning('Mismatched log message: %s' % processed_line)
                        for index, template in self.templates.items():
                            if re.compile(template).match(line) is not None:
                                self.log2temp[log_id] = index
                                break
                        if log_id not in self.log2temp.keys():
                            self.logger.error('Failed to parse line %s' % line)
                            exit(2)
                    log_id += 1

            with open(templates_file, 'w', encoding='utf-8') as writer:
                for id, template in self.templates.items():
                    writer.write(','.join([str(id), template]) + '\n')
            with open(log2temp_file, 'w', encoding='utf-8') as writer:
                for logid, tempid in self.log2temp.items():
                    writer.write(','.join([str(logid), str(tempid)]) + '\n')
            # with open(logseq_file, 'w', encoding='utf-8') as writer:
            #     self._save_log_event_seqs(writer)
        self._prepare_semantic_embed(os.path.join(save_path, 'event2semantic.vec'))
        # Summarize log event sequences.
        for block, seq in self.block2seqs.items():
            self.block2eventseq[block] = []
            for log_id in seq:
                self.block2eventseq[block].append(self.log2temp[log_id])

    def _load_raw_log_seqs(self):
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        label_file = os.path.join(self.dataset_base, 'label.txt')
        if os.path.exists(sequence_file) and os.path.exists(label_file):
            self.logger.info('Start load from previous extraction. File path %s' % sequence_file)
            with open(sequence_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(':')
                    block = tokens[0]
                    seq = tokens[1].split()
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]
            with open(label_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    block_id, label = line.strip().split(':')
                    self.block2label[block_id] = label

        else:
            self.logger.info('Start loading BGL log sequences.')
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                nodes = OrderedDict()
                for idx, line in enumerate(lines):
                    tokens = line.strip().split()
                    node = str(tokens[3]) # 按第3个token聚合
                    if node not in nodes.keys():
                        nodes[node] = []
                    nodes[node].append((idx, line.strip()))

                pbar = tqdm(total=len(nodes))

                block_idx = 0
                for node, seq in nodes.items():
                    if len(seq) < self.window_size:
                        self.blocks.append(str(block_idx))
                        self.block2seqs[str(block_idx)] = []
                        label = 'Normal'
                        for (idx, line) in seq:
                            self.block2seqs[str(block_idx)].append(int(idx))
                            if not line.startswith('-'):
                                label = "Anomaly"
                        self.block2label[str(block_idx)] = label
                        block_idx += 1
                    else:
                        i = 0
                        while i < len(seq):
                            self.blocks.append(str(block_idx))
                            self.block2seqs[str(block_idx)] = []
                            label = 'Normal'
                            for (idx, line) in seq[i:i + self.window_size]:
                                self.block2seqs[str(block_idx)].append(int(idx))
                                if not line.startswith('-'):
                                    label = "Anomaly"
                            self.block2label[str(block_idx)] = label
                            block_idx += 1
                            i += self.window_size

                    pbar.update(1)

                pbar.close()
            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')

            with open(label_file, 'w', encoding='utf-8') as writer:
                for block in self.block2label.keys():
                    writer.write(':'.join([block, self.block2label[block]]) + '\n')

        self.logger.info('Extraction finished successfully.')
        pass


if __name__ == '__main__':
    from representations.templates.statistics import Simple_template_TF_IDF

    semantic_encoder = Simple_template_TF_IDF()
    loader = BGLLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/temp_BGL/BGL.log'),
                       dataset_base=os.path.join(PROJECT_ROOT, 'datasets/temp_BGL'),
                       semantic_repr_func=semantic_encoder.present)
    loader.parse_by_IBM(config_file=os.path.join(PROJECT_ROOT, 'conf/BGL.ini'),
                        persistence_folder=os.path.join(PROJECT_ROOT, 'datasets/temp_BGL/persistences'))
    pass
