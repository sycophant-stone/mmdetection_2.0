import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def research_logger_info(str):
    '''
    wrap logger.info..
    :param str:
    :return:
    '''
    logger.info(str)

if __name__=='__main__':
    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
