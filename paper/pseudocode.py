# pseudocode 

def latq():
    # monitor stream that append new requests
    while True:
        if new request r from client u arrive:
            # when current client does not have a waiting request
            if not req in Q and client(req) == u:
                # update the timestamp of the request
                pass
            else:
                Q.append(r)

    # execution stream that schedule next request 
    while True:
        if can_add_new_request():



# dont need to consider continous batching since inference engine handles batching
