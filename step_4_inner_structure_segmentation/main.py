from train_util import *
from unet import UNet
import sys
sys.path.append(os.path.abspath('..'))
from utils.utils_constants import CROPPED_VESSELS_DIR

def main():
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = UNet()
    net = net.to(device)
    root = os.path.join(CROPPED_VESSELS_DIR, "Arterioles")
    train(device=device, root=root, net=net, epochs=50, batch_size=16, lr=0.001, reg=0.95, log_every_n=10)
    net.load_state_dict(torch.load("net.pt", map_location=device))
    test(device=device, root=root, net=net)

if __name__ == '__main__':
    main()