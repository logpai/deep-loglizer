import numpy as np

perturbed_log_event_dict = {
    "Adding an already existing block <*>": "Adding an existing block <*>",
    "<*>Verification succeeded for <*>": "<*>Verification is successful for <*>",
    "<*> Served block <*> to <*>": "<*> Served block <*> to <*> time <*>",
    "<*>:Got exception while serving <*> to <*>:<*>": "<*>:Got error while serving <*> to <*>:<*>",
    "Receiving block <*> src: <*> dest: <*>": "Receiving block <*> from src<*> to dest<*>",
    "Received block <*> src: <*> dest: <*> of size ([-]?[0-9]+)": "Received block <*> from src<*> to dest<*> of size ([-]?[0-9]+)",
    "writeBlock <*> received exception <*>": "writeBlock <*> got error <*>",
    "PacketResponder ([-]?[0-9]+) for block <*> Interrupted\.": "PacketResponder ([-]?[0-9]+) for block <*> Failed\.",
    "Received block <*> of size ([-]?[0-9]+) from <*>": "Received block <*> from <*> size ([-]?[0-9]+)",
    "PacketResponder <*> ([-]?[0-9]+) Exception <*>": "PacketResponder <*> ([-]?[0-9]+) Error <*>",
    "PacketResponder ([-]?[0-9]+) for block <*> terminating": "PacketResponder ([-]?[0-9]+) for block <*> terminating time (*)",
    "<*>:Exception writing block <*> to mirror <*><*>": "<*>:Error writing block <*> to mirror <*><*>",
    "Receiving empty packet for block <*>": "Receiving empty packet block <*>",
    "Exception in receiveBlock for block <*> <*>": "Error in receiveBlock for block <*> <*>",
    "Changing block file offset of block <*> from ([-]?[0-9]+) to ([-]?[0-9]+) meta file offset to ([-]?[0-9]+)": "Modifying block file offset of block <*> from ([-]?[0-9]+) to ([-]?[0-9]+) meta file offset to ([-]?[0-9]+)",
    "<*>:Transmitted block <*> to <*>": "<*>:Transmitted block <*> to <*> time (*)",
    "<*>:Failed to transfer <*> to <*> got <*>": "<*>:Error happened to transfer <*> to <*> got <*>",
    "<*> Starting thread to transfer block <*> to <*>": "<*> Starting thread to move block <*> to <*>",
    "Reopen Block <*>": "Open Block <*> Again",
    "Unexpected error trying to delete block <*>\. BlockInfo not found in volumeMap\.": "Unexpected exception trying to delete block <*>\. BlockInfo not found in volumeMap\.",
    "Deleting block <*> file <*>": "Trying to delete block <*> file <*>",
    "BLOCK\* NameSystem\.allocateBlock: <*>\. <*>": "BLOCK\* NameSystem\.allocateBlock<*>\. <*> time <*>",
    "BLOCK\* NameSystem\.delete: <*> is added to invalidSet of <*>": "BLOCK\* NameSystem\.delete<*> is added to invalidSet <*>",
    "BLOCK\* Removing block <*> from neededReplications as it does not belong to any file\.": "BLOCK\* Deleting block <*> from neededReplications as it does not belong to any file\.",
    "BLOCK\* ask <*> to replicate <*> to <*>": "BLOCK\* enquire <*> to replicate <*> to <*>",
    "BLOCK\* NameSystem\.addStoredBlock: blockMap updated: <*> is added to <*> size ([-]?[0-9]+)": "BLOCK\* NameSystem\.addStoredBlockblockMap updated<*> is added to <*> on size ([-]?[0-9]+) time <*>",
    "BLOCK\* NameSystem\.addStoredBlock: Redundant addStoredBlock request received for <*> on <*> size ([-]?[0-9]+)": "BLOCK\* NameSystem\.addStoredBlockRedundant addStoredBlock request received for <*> <*> size ([-]?[0-9]+)",
    "BLOCK\* NameSystem\.addStoredBlock: addStoredBlock request received for <*> on <*> size ([-]?[0-9]+) But it does not belong to any file\.": "BLOCK\* NameSystem\.addStoredBlockaddStoredBlock request received for <*> on <*> size ([-]?[0-9]+). However, it does not belong to any file\.",
    "PendingReplicationMonitor timed out block <*>": "PendingReplicationMonitor timed out on block <*> time elapsed <*>",
}


def perturb_hdfs(
    test_log_seqs,
    inject_ratio=0,
    remove_event_num=1,
    duplicate_event_num=5,
    shuffle_subseq_len=5,
    injected_event_num=10,
):
    log_seq_num = len(test_log_seqs)
    seq_idxes = np.arange(log_seq_num)
    np.random.shuffle(seq_idxes)
    # seq_idxes = [idx for idx in seq_idxes if len(test_log_seqs[idx]) >= 5]
    inject_seq_ids = seq_idxes[
        : int(log_seq_num * inject_ratio)
    ]  # the log seqs to inject noise
    perturb_types = np.random.randint(
        4, size=len(inject_seq_ids)
    )  # five perturbation choices

    test_log_seqs = list(test_log_seqs)
    for i in range(len(inject_seq_ids)):
        seq_idx = inject_seq_ids[i]
        test_log_seq = test_log_seqs[seq_idx]
        perturb_type = perturb_types[i]

        # Remove one log event
        if perturb_type == 0:
            for _ in range(remove_event_num):
                event_idx = np.random.randint(len(test_log_seq), size=1)[0]
                test_log_seq = test_log_seq[:event_idx] + test_log_seq[event_idx + 1 :]

        # Duplicate one log event
        elif perturb_type == 1:
            for _ in range(duplicate_event_num):
                event_idx = np.random.randint(len(test_log_seq), size=1)[0]
                test_log_seq = (
                    test_log_seq[:event_idx]
                    + [test_log_seq[event_idx]]
                    + test_log_seq[event_idx:]
                )

        # Shuffle the order of a log subsequence
        elif perturb_type == 2:
            shuffle_len_tmp = (
                shuffle_subseq_len
                if len(test_log_seq) > shuffle_subseq_len
                else len(test_log_seq) - 1
            )
            event_idx = np.random.randint(len(test_log_seq) - shuffle_len_tmp, size=1)[
                0
            ]
            log_subseq = test_log_seq[event_idx : event_idx + shuffle_subseq_len]
            idxes = np.arange(len(log_subseq))
            np.random.shuffle(idxes)
            log_subseq = np.array(log_subseq)[idxes].tolist()
            test_log_seq = (
                test_log_seq[:event_idx]
                + log_subseq
                + test_log_seq[event_idx + shuffle_subseq_len :]
            )

        # Inject pseudo log events
        elif perturb_type == 3:
            event_idxes = np.random.randint(len(test_log_seq), size=injected_event_num)
            for event_idx in event_idxes:
                test_log_seq[event_idx] = perturbed_log_event_dict[
                    test_log_seq[event_idx]
                ]

        test_log_seqs[seq_idx] = list(test_log_seq)

    return np.array(test_log_seqs, dtype=object)
