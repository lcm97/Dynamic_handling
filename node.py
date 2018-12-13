"""
    This module shows the node for 4/6 nodes distributed system setup.
"""
import argparse
import os
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import Model_5, Model_6, Model_7, Model_8
from collections import deque, OrderedDict
from multiprocessing import Queue
from threading import Thread, Lock
import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import tensorflow as tf
import yaml


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# read data packet format.
PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class Node(object):
    """
        Singleton Node class. It will store data if necessary, record next layer
        response time, send data packet to next layer and store the loaded model
        in memory without reloading.

        Attributes:
            ip: A dictionary contains Queue of ip addresses for different models type.
            model: Loaded models associated to a node.
            graph: Default graph used by Tensorflow.
            debug: Flag for debugging.
            lock: Threading lock for safe usage of this class. The lock is used
                    for safe models forwarding. If the models is processing input and
                    it gets request from other devices, the new request will wait
                    until the previous models forwarding finishes.
            name: Model name.
            total: Total time of getting frames.
            count: Total number of frames gets back.
            input: Store the input for last fully connected layer, it acts as a buffer
                    that it will kick out extra data and store unused data.
    """

    instance = None

    def __init__(self):
        self.ip = dict()
        self.model = None
        self.graph = tf.get_default_graph()
        self.debug = False
        self.lock = Lock()
        self.name = 'unknown'
        self.total = 0
        self.count = 1
        self.input = deque()
        self.num_devices = 6

    def log(self, step, data=''):
        """
            Log function for debug. Turn the flag on to show each step result.

            Args:
                step: Each step names.
                data: Data format or size.
        """
        if self.debug:
            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            for k in range(0, len(step), 68):
                print '+{:^68.68}+'.format(step[k:k + 68])
            for k in range(0, len(data), 68):
                print '+{:^68.68}+'.format(data[k:k + 68])
            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print

    def acquire_lock(self):
        self.lock.acquire()

    def release_lock(self):
        self.lock.release()

    def timer(self, interval):
        self.total += interval
        print '{:s}: {:.3f}'.format(self.name, self.total / self.count)
        self.count += 1

    @classmethod
    def create(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


class Responder(ipc.Responder):
    """ Responder called by handler when got request. """

    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """
            This functino is invoked by do_POST to handle the request. Invoke handles
            the request and get response for the request. This is the key of each node.
            All models forwarding and output redirect are done here. Because the invoke
            method of initializer only needs to receive the data packet, it does not do
            anything in the function and return None.

            Because this is a node class, it has all necessary code here for handling
            different inputs. Basically the logic is load model as the previous layer
            request and run model inference. And it will send the current layer output
            to next layer. We write different model's code all here for the sake of
            convenience. In order to avoid long waiting time of model reloading, we
            make sure each node is assigned to a unique job each time, so it does not
            need to reload the model.

            Args:
                msg: Meta data.
                req: Contains data packet.

            Returns:
                None: It just acts as confirmation for sender.

            Raises:
                AvroException: if the data does not have correct syntac defined in Schema
        """
        node = Node.create()
        node.acquire_lock()

        if msg.name == 'forward':
            try:
                with node.graph.as_default():
                    if node.num_devices == 5:
                        output, name = Model_5.forward(req['input'], req['next'], node)
                        Thread(target=self.send, args=(output, name, req['tag'])).start()
                    elif node.num_devices == 6:
                        output, name = Model_6.forward(req['input'], req['next'], node)
                        Thread(target=self.send, args=(output, name, req['tag'])).start()
                    elif node.num_devices == 7:
                        output, name = Model_7.forward(req['input'], req['next'], node)
                        Thread(target=self.send, args=(output, name, req['tag'])).start()
                    elif node.num_devices == 8:
                        output, name = Model_8.forward(req['input'], req['next'], node)
                        Thread(target=self.send, args=(output, name, req['tag'])).start()

                node.release_lock()
                return

            except Exception, e:
                node.log('Error', e.message)
        else:
            raise schema.AvroException('unexpected message:', msg.getname())

    def send(self, X, name, tag):
        """
            Send data to other devices. The data packet contains data and models name.
            Ip address of next device pop from Queue of a ip list.

            Args:
                 X: numpy array
                 name: next device models name
                 tag: mark the current layer label
        """
        node = Node.create()
        queue = node.ip[name]
        address = queue.get()

        # initializer use port 9999 to receive data
        port = 9999 if name == 'initial' else 12345
        client = ipc.HTTPTransceiver(address, port)
        requestor = ipc.Requestor(PROTOCOL, client)

        node.name = name

        data = dict()
        data['input'] = X.tostring()
        data['next'] = name
        data['tag'] = tag
        node.log('finish assembly')
        start = time.time()
        try:
            requestor.request('forward', data)
        except Exception, e:
            # node.log('Error', e.message)
            # The interrupt node's ip is the address above
            print address
            """Remove the IP address of the interrupted node from the available ip"""
            available_ip = read_ip(get_file(node.num_devices))
            available_ip = del_ip(available_ip, address)

            node.num_devices = node.num_devices - 1
            """Update new IP configuration based on available ip"""
            update_ip(get_file(node.num_devices), available_ip)

        end = time.time()
        node.timer(end - start)
        node.log('node gets request back')

        client.close()
        queue.put(address)


def get_file(num_devices):
    """Get the specified ip file"""
    ip_path = 'resource/ips/'
    ip_file = ip_path + str(num_devices) + '_ip'
    return ip_file


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Load dict into OrderedDict type"""
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def read_ip(ip_file):
    """Read the ip from ip_files"""
    with open(ip_file, "r") as file:
        address = ordered_load(file)
        """address is a dict type object"""
        # print address
        address = address['node']

        available_ip = []
        """Get all the ip addresses"""
        for i in address:
            print i
            for ip in address[i]:
                available_ip.append(ip)
            # print address[i]
        # print address['block1']
        return available_ip


def del_ip(ip_list, interrupted_ip):
    """Delete interrupted ip address"""
    ip_index = ip_list.index(interrupted_ip)
    del ip_list[ip_index]
    return ip_list


def update_ip(ip_file, available_ip):
    """Upgrade the ip configuration and write it to the ip file"""
    with open(ip_file, "r") as file:
        address = ordered_load(file)
        """address is a dict type object"""
        # print address
        address = address['node']

    with open(ip_file, "w") as file:
        print address
        # address['initial'][0] = available_ip[2]
        """Fill the new configuration with available_ip"""
        index = 0
        for i in address:
            for ip in address[i]:
                j = address[i].index(ip)
                address[i][j] = available_ip[index]
                index = index + 1

        print address
        # print address['initial']
        # address = dict(address)
        node_ip = dict()
        node_ip['node'] = address
        yaml.dump(node_ip, file)


def load_ip(node):
    with open(get_file(node.num_devices)) as file:
        """read ip resources from config file"""
        address = yaml.safe_load(file)
        address = address['node']
        for i in address:
            node.ip[i] = Queue()
            for addr in address[i]:
                if addr == '#':
                    break
                node.ip[i].put(addr)
   
                
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """
            do_POST is automatically called by ThreadedHTTPServer. It creates a new
            responder for each request. The responder generates response and write
            response to data sent back.
        """
        self.responder = Responder()
        call_request_reader = ipc.FramedReader(self.rfile)
        call_request = call_request_reader.read_framed_message()
        resp_body = self.responder.respond(call_request)
        self.send_response(200)
        self.send_header('Content-Type', 'avro/binary')
        self.end_headers()
        resp_writer = ipc.FramedWriter(self.wfile)
        resp_writer.write_framed_message(resp_body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """ Handle requests in separate thread. """


def main(cmd):
    node = Node.create()

    node.debug = cmd.debug
    
    load_ip(node)

    server = ThreadedHTTPServer(('0.0.0.0', 12345), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='set to debug mode')
    cmd = parser.parse_args()
    main(cmd)
